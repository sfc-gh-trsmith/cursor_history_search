# Cursor History Search

A CLI tool to search across all your Cursor IDE chat history using semantic search. Find that prompt you wrote weeks ago by describing what you were working on, not by remembering exact keywords.

## Features

- **Semantic Search** — Find prompts by meaning, not just keywords. Ask "how did I set up authentication?" and find relevant prompts even if they don't contain those exact words.
- **Hybrid Ranking** — Combines semantic similarity with lexical matching for best results.
- **Context Display** — Show surrounding prompts (like `grep -C`) to see the conversation flow.
- **Project Filtering** — Filter results to specific projects.
- **Session Viewing** — Browse all prompts from a single coding session.
- **Incremental Indexing** — Only re-indexes changed workspaces for fast updates.
- **Web UI** — Beautiful Streamlit interface for interactive searching.
- **CLI Interface** — Full-featured command line for scripting and quick searches.

## Installation

### Prerequisites

- Python 3.9+ (tested with 3.11)
- macOS (Windows/Linux support coming soon)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install faiss-cpu sentence-transformers filelock streamlit
```

### Download the Script

```bash
# Clone the repository
git clone https://github.com/sfc-gh-trsmith/cursor_history_search.git
cd cursor_history_search

# Or just download the script
curl -O https://raw.githubusercontent.com/sfc-gh-trsmith/cursor_history_search/main/cursor_history_search.py
```

## Quick Start

```bash
# 1. Build the search index (run this first, takes ~1 minute)
python cursor_history_search.py index

# 2. Search for prompts
python cursor_history_search.py search "database migration"

# 3. Or launch the web UI
python cursor_history_search.py --server
```

## Commands Reference

### `index` — Build or Update the Search Index

```bash
# Incremental update (only changed workspaces)
python cursor_history_search.py index

# Force full rebuild
python cursor_history_search.py index --force
```

The index is stored in `~/.cursor_history_index/` and includes:
- `index.faiss` — FAISS vector index
- `metadata.json` — Prompt text and metadata
- `workspace_state.json` — Tracks which workspaces have been indexed

### `search` — Search for Prompts

```bash
# Basic search
python cursor_history_search.py search "how to deploy"

# Get more results
python cursor_history_search.py search "SQL query" --top 20

# Filter by project name
python cursor_history_search.py search "authentication" --project myapp

# Show context (2 prompts before/after each match)
python cursor_history_search.py search "error handling" -C 2
```

**Options:**
| Flag | Description |
|------|-------------|
| `--top N`, `-n N` | Number of results (default: 5) |
| `--project NAME`, `-p NAME` | Filter by project name (partial match) |
| `--context N`, `-C N` | Show N prompts before/after each match |

### `list-projects` — List All Indexed Projects

```bash
python cursor_history_search.py list-projects
```

Shows all projects with prompt counts, sorted by most prompts.

### `stats` — Show Index Statistics

```bash
python cursor_history_search.py stats
```

Displays:
- Total prompts indexed
- Number of unique projects
- Workspaces scanned
- Average prompt length
- Index file sizes

### `--server` — Launch Web UI

```bash
# Default port (8501)
python cursor_history_search.py --server

# Custom port
python cursor_history_search.py --server --port 8080
```

Open `http://localhost:8501` in your browser.

## Web UI

The Streamlit web interface provides:

- **Interactive Search** — Type queries and see results instantly
- **Project Dropdown** — Filter by project
- **Adjustable Results** — Choose how many results to show
- **Context Toggle** — Show surrounding prompts
- **Session Viewer** — Click "View Session" to see all prompts from that coding session
- **Re-index Button** — Rebuild the index from the UI
- **Statistics Sidebar** — See index stats and project list

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cursor IDE Workspaces                        │
│  ~/Library/Application Support/Cursor/User/workspaceStorage/    │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│   │ workspace_1/ │  │ workspace_2/ │  │ workspace_N/ │          │
│   │ state.vscdb  │  │ state.vscdb  │  │ state.vscdb  │          │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└──────────┼─────────────────┼─────────────────┼──────────────────┘
           │                 │                 │
           └────────────────┬┴─────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Extract Prompts       │
              │   from SQLite DBs       │
              │   (aiService.prompts)   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Generate Embeddings   │
              │   sentence-transformers │
              │   (all-MiniLM-L6-v2)    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Build FAISS Index     │
              │   (cosine similarity)   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   ~/.cursor_history_    │
              │   index/                │
              │   ├── index.faiss       │
              │   ├── metadata.json     │
              │   └── workspace_state   │
              └─────────────────────────┘
```

### Search Process

1. **Query Embedding** — Your search query is converted to a vector using the same embedding model
2. **FAISS Retrieval** — Top candidates are retrieved using cosine similarity
3. **Hybrid Re-ranking** — Results are re-scored combining:
   - Semantic similarity (50%)
   - Exact substring match bonus (30%)
   - Word overlap score (20%)
4. **Context Enrichment** — If requested, neighboring prompts are fetched

### Data Storage

- **Source**: Cursor stores chat history in SQLite databases at `~/Library/Application Support/Cursor/User/workspaceStorage/*/state.vscdb`
- **Key**: Prompts are stored under the `aiService.prompts` key in the `ItemTable`
- **Index**: The tool creates its own index at `~/.cursor_history_index/`

The tool opens databases in **read-only mode** and never modifies Cursor's data.

## Platform Support

| Platform | Status | Cursor Data Path |
|----------|--------|------------------|
| macOS | Supported | `~/Library/Application Support/Cursor/User/workspaceStorage/` |
| Windows | Coming Soon | `%APPDATA%\Cursor\User\workspaceStorage\` |
| Linux | Coming Soon | `~/.config/Cursor/User/workspaceStorage/` |

## Troubleshooting

### "No index found" error

Run the index command first:
```bash
python cursor_history_search.py index
```

### "Could not acquire lock" error

Another indexing process may be running. If you're sure it's not, delete the lock file:
```bash
rm ~/.cursor_history_index/index.lock
```

### "Missing required dependency" error

Install all dependencies:
```bash
pip install faiss-cpu sentence-transformers filelock streamlit
```

### Slow first search

The first search loads the embedding model into memory (~100MB). Subsequent searches are fast.

### Index seems outdated

Force a full rebuild:
```bash
python cursor_history_search.py index --force
```

### Database locked warnings

If you see warnings about databases being locked, it means Cursor is actively using them. The tool skips these and indexes what it can.

## Configuration

The tool uses sensible defaults but stores files at:

| File | Location | Purpose |
|------|----------|---------|
| FAISS Index | `~/.cursor_history_index/index.faiss` | Vector search index |
| Metadata | `~/.cursor_history_index/metadata.json` | Prompt text and info |
| Workspace State | `~/.cursor_history_index/workspace_state.json` | Tracks indexed workspaces |
| Lock File | `~/.cursor_history_index/index.lock` | Prevents concurrent indexing |

### Embedding Model

The tool uses `all-MiniLM-L6-v2` from sentence-transformers:
- 384-dimensional embeddings
- Good balance of speed and quality
- ~100MB model size

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Some ideas for enhancements:

- Windows/Linux support
- Index AI responses (not just prompts)
- Date/time filtering
- Export to JSON/CSV/Markdown
- Configuration file support
- Auto-reindex daemon

---

Built for developers who pair-program with AI and want to learn from their past conversations.
