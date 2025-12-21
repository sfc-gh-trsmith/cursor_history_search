#!/usr/bin/env python3
"""
Cursor History Search Tool

A CLI tool to search across all Cursor IDE chat history using semantic search.
Extracts prompts from Cursor's SQLite databases, generates embeddings with
sentence-transformers, and indexes with FAISS for fast similarity search.

Usage:
    python cursor_history_search.py index           # Build/update the index
    python cursor_history_search.py search "query"  # Search for prompts
    python cursor_history_search.py list-projects   # List indexed projects
    python cursor_history_search.py stats           # Show index statistics
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import unquote, urlparse

# Third-party imports (check availability)
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from filelock import FileLock, Timeout
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("\nInstall dependencies with:")
    print("  pip install faiss-cpu sentence-transformers filelock")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Cursor storage paths
CURSOR_BASE_PATH = Path.home() / "Library" / "Application Support" / "Cursor" / "User"
WORKSPACE_STORAGE_PATH = CURSOR_BASE_PATH / "workspaceStorage"

# Index storage location
INDEX_DIR = Path.home() / ".cursor_history_index"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
WORKSPACE_STATE_PATH = INDEX_DIR / "workspace_state.json"
LOCK_PATH = INDEX_DIR / "index.lock"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Lock timeout in seconds
LOCK_TIMEOUT = 10

# Streamlit app temp file
STREAMLIT_APP_PATH = INDEX_DIR / "_streamlit_app.py"


# =============================================================================
# Embedded Streamlit App Code
# =============================================================================

STREAMLIT_APP_CODE = '''
"""Cursor History Search - Streamlit UI"""
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

# Configuration - must match main script
INDEX_DIR = Path.home() / ".cursor_history_index"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
WORKSPACE_STATE_PATH = INDEX_DIR / "workspace_state.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Import heavy dependencies
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize session state
if "view_session" not in st.session_state:
    st.session_state.view_session = None
if "highlight_prompt_idx" not in st.session_state:
    st.session_state.highlight_prompt_idx = None

# Cache the embedding model
@st.cache_resource
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# Cache the index loading
@st.cache_resource
def load_index():
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, [], {}
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        workspace_states = {}
        if WORKSPACE_STATE_PATH.exists():
            with open(WORKSPACE_STATE_PATH, "r") as f:
                workspace_states = json.load(f)
        return index, metadata, workspace_states
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None, [], {}

def build_context_index(metadata):
    """Build index for fast context lookup."""
    ctx_idx = {}
    for p in metadata:
        key = (p["workspace_hash"], p["prompt_index"])
        ctx_idx[key] = p
    return ctx_idx

def get_session_prompts(workspace_hash, metadata):
    """Get all prompts from a workspace session, sorted by prompt_index."""
    session_prompts = [p for p in metadata if p["workspace_hash"] == workspace_hash]
    return sorted(session_prompts, key=lambda x: x["prompt_index"])

def get_prompt_context(result, context_idx, context_size=2):
    """Get neighboring prompts for context."""
    ws_hash = result["workspace_hash"]
    prompt_idx = result["prompt_index"]
    
    before = []
    after = []
    
    for i in range(context_size, 0, -1):
        key = (ws_hash, prompt_idx - i)
        if key in context_idx:
            before.append(context_idx[key])
    
    for i in range(1, context_size + 1):
        key = (ws_hash, prompt_idx + i)
        if key in context_idx:
            after.append(context_idx[key])
    
    return {"before": before, "match": result, "after": after}

def compute_rerank_score(query, text, semantic_score):
    """Compute hybrid score combining semantic + lexical matching."""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Exact substring match bonus
    exact_match_bonus = 0.5 if query_lower in text_lower else 0.0
    
    # Word overlap score
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    word_overlap_score = len(query_words & text_words) / len(query_words) if query_words else 0.0
    
    # Combined: 50% semantic, 30% exact match, 20% word overlap
    return 0.5 * semantic_score + 0.3 * exact_match_bonus + 0.2 * word_overlap_score

def search_prompts(query, top_k=10, project_filter=None, context_size=0):
    index, metadata, _ = load_index()
    if index is None or not metadata:
        return []
    
    model = get_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    
    # Stage 1: Retrieve more candidates for re-ranking
    candidate_k = max(50, top_k * 5)
    if project_filter and project_filter != "All":
        candidate_k *= 3
    
    distances, indices = index.search(query_embedding, min(candidate_k, len(metadata)))
    
    context_idx = build_context_index(metadata) if context_size > 0 else None
    
    # Stage 2: Re-rank with hybrid scoring
    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        prompt = metadata[idx].copy()
        semantic_score = float(dist)
        
        if project_filter and project_filter != "All":
            if project_filter.lower() not in prompt["project"].lower():
                continue
        
        # Hybrid score
        combined_score = compute_rerank_score(query, prompt["text"], semantic_score)
        prompt["semantic_score"] = semantic_score
        prompt["score"] = combined_score
        candidates.append(prompt)
    
    # Sort by combined score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    results = candidates[:top_k]
    
    # Add context
    if context_size > 0:
        for prompt in results:
            prompt["context"] = get_prompt_context(prompt, context_idx, context_size)
    
    return results

def get_stats():
    index, metadata, workspace_states = load_index()
    if index is None:
        return None
    
    projects = {}
    for p in metadata:
        proj = p["project"]
        projects[proj] = projects.get(proj, 0) + 1
    
    faiss_size = FAISS_INDEX_PATH.stat().st_size if FAISS_INDEX_PATH.exists() else 0
    meta_size = METADATA_PATH.stat().st_size if METADATA_PATH.exists() else 0
    avg_len = sum(len(p["text"]) for p in metadata) / len(metadata) if metadata else 0
    
    return {
        "total_prompts": len(metadata),
        "unique_projects": len(projects),
        "workspaces_scanned": len(workspace_states),
        "avg_prompt_length": avg_len,
        "faiss_size_kb": faiss_size / 1024,
        "meta_size_kb": meta_size / 1024,
        "projects": dict(sorted(projects.items(), key=lambda x: x[1], reverse=True)),
    }

def run_reindex():
    """Run reindex in subprocess."""
    script_path = Path(__file__).parent / "cursor_history_search.py"
    if not script_path.exists():
        script_path = INDEX_DIR.parent / "cursor_history_search.py"
    
    result = subprocess.run(
        [sys.executable, str(script_path), "index", "--force"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr

# Page config
st.set_page_config(
    page_title="Cursor History Search",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS - dark mode compatible
st.markdown("""
<style>
    .context-prompt {
        background-color: rgba(100, 100, 100, 0.2);
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.9em;
        border-left: 2px solid #888;
    }
    .match-prompt {
        background-color: rgba(76, 175, 80, 0.2);
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #4CAF50;
        font-weight: 500;
    }
    .session-prompt {
        background-color: rgba(33, 150, 243, 0.1);
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #2196F3;
    }
    .session-prompt-highlight {
        background-color: rgba(255, 193, 7, 0.35);
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        border-left: 6px solid #ffc107;
        border: 2px solid #ffc107;
        font-weight: 500;
        box-shadow: 0 0 10px rgba(255, 193, 7, 0.4);
    }
    .context-label {
        font-size: 0.75em;
        color: #aaa;
        font-weight: bold;
        margin-right: 8px;
    }
    .prompt-index {
        font-size: 0.8em;
        color: #ccc;
        background-color: rgba(100, 100, 100, 0.3);
        padding: 2px 6px;
        border-radius: 3px;
        margin-right: 8px;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 2em;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 Cursor History Search")
    st.markdown("---")
    
    stats = get_stats()
    if stats:
        st.subheader("📊 Index Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompts", f"{stats['total_prompts']:,}")
            st.metric("Projects", stats["unique_projects"])
        with col2:
            st.metric("Workspaces", stats["workspaces_scanned"])
            st.metric("Avg Length", f"{stats['avg_prompt_length']:.0f}")
        
        st.caption(f"Index: {stats['faiss_size_kb']:.1f} KB | Meta: {stats['meta_size_kb']:.1f} KB")
        st.markdown("---")
        
        st.subheader("📁 Projects")
        for proj, count in list(stats["projects"].items())[:15]:
            st.text(f"{proj[:25]:<25} {count:>4}")
        if len(stats["projects"]) > 15:
            st.caption(f"... and {len(stats['projects']) - 15} more")
    else:
        st.warning("No index found. Click Re-index to build.")
    
    st.markdown("---")
    
    if st.button("🔄 Re-index", use_container_width=True):
        with st.spinner("Re-indexing... This may take a minute."):
            success, output = run_reindex()
            if success:
                st.success("Re-index complete!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Re-index failed")
                st.code(output)

# ============================================================================
# Session View Mode
# ============================================================================
if st.session_state.view_session:
    session_info = st.session_state.view_session
    ws_hash = session_info["workspace_hash"]
    project = session_info["project"]
    highlight_idx = st.session_state.highlight_prompt_idx
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Search"):
            st.session_state.view_session = None
            st.session_state.highlight_prompt_idx = None
            st.rerun()
    
    st.title(f"Session History: `{project}`")
    st.caption(f"All prompts from this agent session (workspace: {ws_hash[:12]}...)")
    
    # Get all session prompts
    _, metadata, _ = load_index()
    session_prompts = get_session_prompts(ws_hash, metadata)
    
    st.info(f"📜 **{len(session_prompts)} prompts** in this session • Scrolled to matched prompt")
    
    # Display all prompts
    for prompt in session_prompts:
        idx = prompt["prompt_index"]
        is_highlight = (idx == highlight_idx)
        
        css_class = "session-prompt-highlight" if is_highlight else "session-prompt"
        label = "⭐ MATCHED PROMPT" if is_highlight else f"#{idx}"
        
        # Add id for scrolling to highlighted prompt
        element_id = f'id="highlight-target"' if is_highlight else ""
        
        st.markdown(
            f'<div {element_id} class="{css_class}">'
            f'<span class="prompt-index">{label}</span>'
            f'{prompt["text"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # JavaScript to scroll to highlighted element
    if highlight_idx is not None:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                setTimeout(function() {
                    var element = window.parent.document.getElementById('highlight-target');
                    if (element) {
                        element.scrollIntoView({behavior: 'smooth', block: 'center'});
                    }
                }, 300);
            </script>
            """,
            height=0,
        )

# ============================================================================
# Search View Mode (default)
# ============================================================================
else:
    st.title("Cursor History Search")
    st.caption("Semantic search across all your Cursor IDE chat history")
    
    # Search controls
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="Enter your search query...",
            label_visibility="collapsed",
        )
    
    with col2:
        stats = get_stats()
        project_options = ["All"] + (list(stats["projects"].keys()) if stats else [])
        project_filter = st.selectbox("Project", project_options, label_visibility="collapsed")
    
    with col3:
        top_k = st.selectbox("Results", [5, 10, 20, 50], index=1, label_visibility="collapsed")
    
    with col4:
        context_size = st.selectbox("Context", [0, 1, 2, 3, 5], index=0, 
                                     format_func=lambda x: f"±{x}" if x > 0 else "None",
                                     label_visibility="collapsed")
    
    # Search and display results
    if query:
        with st.spinner("Searching..."):
            results = search_prompts(
                query, 
                top_k=top_k, 
                project_filter=project_filter if project_filter != "All" else None,
                context_size=context_size
            )
        
        if results:
            st.success(f"Found {len(results)} matches for: **{query}**")
            
            for i, result in enumerate(results, 1):
                # Result header with View Session button
                col_header, col_btn = st.columns([5, 1])
                with col_header:
                    st.markdown(f"### [{i}] Score: {result['score']:.2f} — `{result['project']}`")
                with col_btn:
                    if st.button("View Session", key=f"session_{i}"):
                        st.session_state.view_session = {
                            "workspace_hash": result["workspace_hash"],
                            "project": result["project"],
                        }
                        st.session_state.highlight_prompt_idx = result["prompt_index"]
                        st.rerun()
                
                # Show context if available
                if context_size > 0 and "context" in result:
                    ctx = result["context"]
                    
                    for j, before_prompt in enumerate(ctx["before"]):
                        offset = -(len(ctx["before"]) - j)
                        text = before_prompt["text"][:200] + "..." if len(before_prompt["text"]) > 200 else before_prompt["text"]
                        st.markdown(f'<div class="context-prompt"><span class="context-label">[{offset:+d}]</span>{text}</div>', 
                                   unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="match-prompt"><span class="context-label">[MATCH]</span>{result["text"]}</div>', 
                               unsafe_allow_html=True)
                    
                    for j, after_prompt in enumerate(ctx["after"], 1):
                        text = after_prompt["text"][:200] + "..." if len(after_prompt["text"]) > 200 else after_prompt["text"]
                        st.markdown(f'<div class="context-prompt"><span class="context-label">[+{j}]</span>{text}</div>', 
                                   unsafe_allow_html=True)
                else:
                    st.code(result["text"], language=None)
                
                st.markdown("---")
        else:
            st.warning(f"No matches found for: **{query}**")
    else:
        st.info("👆 Enter a search query above to find similar prompts from your Cursor history.")
        
        if stats:
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['total_prompts']:,}</div>
                    <div class="stat-label">Total Prompts</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['unique_projects']}</div>
                    <div class="stat-label">Projects</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['workspaces_scanned']}</div>
                    <div class="stat-label">Workspaces</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['avg_prompt_length']:.0f}</div>
                    <div class="stat-label">Avg Chars</div>
                </div>
                """, unsafe_allow_html=True)
'''


# =============================================================================
# Prompt Extraction
# =============================================================================

def get_project_name_from_workspace(workspace_path: Path) -> Optional[str]:
    """Extract project folder name from workspace.json."""
    workspace_json = workspace_path / "workspace.json"
    if not workspace_json.exists():
        return None
    
    try:
        with open(workspace_json, 'r') as f:
            data = json.load(f)
        
        folder_uri = data.get("folder", "")
        if folder_uri.startswith("file://"):
            # Parse the file URI and extract path
            parsed = urlparse(folder_uri)
            folder_path = unquote(parsed.path)
            return Path(folder_path).name
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def extract_prompts_from_workspace(workspace_path: Path) -> List[Dict[str, Any]]:
    """
    Extract prompts from a workspace's state.vscdb SQLite database.
    Opens database in read-only mode to ensure no modifications.
    
    Returns list of prompt dictionaries with text and metadata.
    """
    db_path = workspace_path / "state.vscdb"
    if not db_path.exists():
        return []
    
    prompts = []
    project_name = get_project_name_from_workspace(workspace_path) or workspace_path.name
    
    try:
        # Open in read-only mode using URI
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        # Query the aiService.prompts key from ItemTable
        cursor.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            ("aiService.prompts",)
        )
        row = cursor.fetchone()
        
        if row and row[0]:
            try:
                # Parse JSON array of prompts
                prompt_data = json.loads(row[0])
                if isinstance(prompt_data, list):
                    for idx, item in enumerate(prompt_data):
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"].strip()
                            if text:  # Skip empty prompts
                                prompts.append({
                                    "text": text,
                                    "project": project_name,
                                    "workspace_hash": workspace_path.name,
                                    "command_type": item.get("commandType", 0),
                                    "prompt_index": idx,
                                })
            except json.JSONDecodeError:
                pass
        
        conn.close()
    except sqlite3.Error as e:
        # Database might be locked or corrupted - skip it
        print(f"  Warning: Could not read {db_path.name}: {e}", file=sys.stderr)
    
    return prompts


def extract_all_prompts() -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
    """
    Extract prompts from all Cursor workspaces.
    
    Returns:
        Tuple of (all_prompts, workspace_states) where workspace_states
        maps workspace_hash to {mtime, prompt_count}
    """
    if not WORKSPACE_STORAGE_PATH.exists():
        print(f"Error: Cursor workspace storage not found at {WORKSPACE_STORAGE_PATH}")
        sys.exit(1)
    
    all_prompts = []
    workspace_states = {}
    
    # Iterate through all workspace directories
    workspace_dirs = [d for d in WORKSPACE_STORAGE_PATH.iterdir() if d.is_dir()]
    
    print(f"Scanning {len(workspace_dirs)} workspaces...")
    
    for workspace_path in workspace_dirs:
        db_path = workspace_path / "state.vscdb"
        if not db_path.exists():
            continue
        
        # Get file modification time
        mtime = db_path.stat().st_mtime
        
        # Extract prompts
        prompts = extract_prompts_from_workspace(workspace_path)
        
        if prompts:
            project_name = prompts[0]["project"]
            print(f"  Found {len(prompts)} prompts in {project_name}")
            all_prompts.extend(prompts)
        
        # Track workspace state
        workspace_states[workspace_path.name] = {
            "mtime": mtime,
            "prompt_count": len(prompts),
            "project": prompts[0]["project"] if prompts else None,
        }
    
    return all_prompts, workspace_states


# =============================================================================
# Embedding Generation
# =============================================================================

_model = None

def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype('float32')


# =============================================================================
# FAISS Index Management
# =============================================================================

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index from embeddings using inner product (cosine similarity)."""
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index using inner product (equivalent to cosine similarity after normalization)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    return index


def save_index(index: faiss.Index, metadata: List[Dict], workspace_states: Dict):
    """Save FAISS index and metadata atomically."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write to temp files first, then rename for atomicity
    with tempfile.NamedTemporaryFile(delete=False, dir=INDEX_DIR) as tmp:
        tmp_faiss = Path(tmp.name)
    with tempfile.NamedTemporaryFile(delete=False, dir=INDEX_DIR, mode='w') as tmp:
        tmp_meta = Path(tmp.name)
    with tempfile.NamedTemporaryFile(delete=False, dir=INDEX_DIR, mode='w') as tmp:
        tmp_state = Path(tmp.name)
    
    try:
        # Write FAISS index
        faiss.write_index(index, str(tmp_faiss))
        
        # Write metadata
        with open(tmp_meta, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Write workspace state
        with open(tmp_state, 'w') as f:
            json.dump(workspace_states, f, indent=2)
        
        # Atomic rename
        shutil.move(str(tmp_faiss), str(FAISS_INDEX_PATH))
        shutil.move(str(tmp_meta), str(METADATA_PATH))
        shutil.move(str(tmp_state), str(WORKSPACE_STATE_PATH))
        
    except Exception:
        # Clean up temp files on failure
        for p in [tmp_faiss, tmp_meta, tmp_state]:
            if p.exists():
                p.unlink()
        raise


def load_index() -> Tuple[Optional[faiss.Index], List[Dict], Dict]:
    """Load existing FAISS index and metadata."""
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, [], {}
    
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        workspace_states = {}
        if WORKSPACE_STATE_PATH.exists():
            with open(WORKSPACE_STATE_PATH, 'r') as f:
                workspace_states = json.load(f)
        
        return index, metadata, workspace_states
    
    except Exception as e:
        print(f"Warning: Could not load existing index: {e}", file=sys.stderr)
        return None, [], {}


# =============================================================================
# Incremental Indexing
# =============================================================================

def get_changed_workspaces(current_states: Dict, saved_states: Dict) -> Tuple[List[str], List[str]]:
    """
    Compare current workspace states with saved states to find changes.
    
    Returns:
        Tuple of (new_or_modified_hashes, removed_hashes)
    """
    new_or_modified = []
    removed = []
    
    # Find new or modified workspaces
    for ws_hash, current in current_states.items():
        saved = saved_states.get(ws_hash)
        if saved is None or saved.get("mtime") != current.get("mtime"):
            new_or_modified.append(ws_hash)
    
    # Find removed workspaces
    for ws_hash in saved_states:
        if ws_hash not in current_states:
            removed.append(ws_hash)
    
    return new_or_modified, removed


def incremental_index(force_full: bool = False) -> Tuple[faiss.Index, List[Dict], Dict]:
    """
    Build or incrementally update the index.
    
    Args:
        force_full: If True, rebuild the entire index from scratch
    
    Returns:
        Tuple of (index, metadata, workspace_states)
    """
    # Load existing index if available
    existing_index, existing_metadata, saved_states = load_index()
    
    if force_full or existing_index is None:
        # Full rebuild
        print("Building full index...")
        all_prompts, workspace_states = extract_all_prompts()
        
        if not all_prompts:
            print("No prompts found to index.")
            return None, [], {}
        
        print(f"\nGenerating embeddings for {len(all_prompts)} prompts...")
        texts = [p["text"] for p in all_prompts]
        embeddings = generate_embeddings(texts)
        
        print("Building FAISS index...")
        index = create_faiss_index(embeddings)
        
        return index, all_prompts, workspace_states
    
    # Incremental update
    print("Checking for changes...")
    
    # Get current workspace states
    _, current_states = extract_all_prompts()
    
    new_or_modified, removed = get_changed_workspaces(current_states, saved_states)
    
    if not new_or_modified and not removed:
        print("No changes detected. Index is up to date.")
        return existing_index, existing_metadata, saved_states
    
    print(f"Found {len(new_or_modified)} new/modified workspaces, {len(removed)} removed")
    
    # For simplicity, if there are changes, rebuild the affected parts
    # A more sophisticated approach would use IndexIDMap for selective updates
    
    # Filter out prompts from removed or modified workspaces
    kept_prompts = [
        p for p in existing_metadata
        if p["workspace_hash"] not in new_or_modified and p["workspace_hash"] not in removed
    ]
    
    # Extract prompts from new/modified workspaces
    new_prompts = []
    for ws_hash in new_or_modified:
        ws_path = WORKSPACE_STORAGE_PATH / ws_hash
        if ws_path.exists():
            prompts = extract_prompts_from_workspace(ws_path)
            new_prompts.extend(prompts)
    
    # Combine
    all_prompts = kept_prompts + new_prompts
    
    if not all_prompts:
        print("No prompts remaining after update.")
        return None, [], current_states
    
    print(f"\nRe-generating embeddings for {len(all_prompts)} prompts...")
    texts = [p["text"] for p in all_prompts]
    embeddings = generate_embeddings(texts)
    
    print("Rebuilding FAISS index...")
    index = create_faiss_index(embeddings)
    
    return index, all_prompts, current_states


# =============================================================================
# Search
# =============================================================================

def compute_rerank_score(query: str, text: str, semantic_score: float) -> float:
    """
    Compute a combined score using semantic similarity and lexical matching.
    
    Combines:
    - Semantic score from FAISS (cosine similarity)
    - Exact match bonus (if query appears as substring)
    - Word overlap score (what fraction of query words appear in text)
    """
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Exact substring match - big bonus
    exact_match_bonus = 0.0
    if query_lower in text_lower:
        exact_match_bonus = 0.5  # Significant boost for exact matches
    
    # Word overlap score
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    
    if query_words:
        # What fraction of query words appear in the text?
        overlap = len(query_words & text_words)
        word_overlap_score = overlap / len(query_words)
    else:
        word_overlap_score = 0.0
    
    # Combined score: weighted combination
    # semantic_score is typically 0-1 (cosine similarity)
    # Give semantic 50% weight, exact match 30%, word overlap 20%
    combined = (
        0.5 * semantic_score +
        0.3 * exact_match_bonus +
        0.2 * word_overlap_score
    )
    
    return combined


def build_context_index(metadata: List[Dict]) -> Dict[Tuple[str, int], Dict]:
    """
    Build an index for fast context lookup.
    
    Returns:
        Dict mapping (workspace_hash, prompt_index) to prompt data
    """
    context_idx = {}
    for prompt in metadata:
        key = (prompt["workspace_hash"], prompt["prompt_index"])
        context_idx[key] = prompt
    return context_idx


def get_prompt_context(
    result: Dict,
    metadata: List[Dict],
    context_size: int = 2,
    context_idx: Optional[Dict] = None,
) -> Dict:
    """
    Get neighboring prompts for context around a search result.
    
    Args:
        result: The matched prompt
        metadata: Full metadata list
        context_size: Number of prompts before/after to include
        context_idx: Pre-built context index (optional, for efficiency)
    
    Returns:
        Dict with 'before', 'match', and 'after' keys
    """
    if context_idx is None:
        context_idx = build_context_index(metadata)
    
    ws_hash = result["workspace_hash"]
    prompt_idx = result["prompt_index"]
    
    before = []
    after = []
    
    # Get prompts before
    for i in range(context_size, 0, -1):
        key = (ws_hash, prompt_idx - i)
        if key in context_idx:
            before.append(context_idx[key])
    
    # Get prompts after
    for i in range(1, context_size + 1):
        key = (ws_hash, prompt_idx + i)
        if key in context_idx:
            after.append(context_idx[key])
    
    return {
        "before": before,
        "match": result,
        "after": after,
    }


def search_prompts(
    query: str,
    top_k: int = 5,
    project_filter: Optional[str] = None,
    context_size: int = 0,
) -> List[Dict]:
    """
    Search for prompts similar to the query using hybrid ranking.
    
    Uses a two-stage approach:
    1. FAISS retrieves top candidates using semantic similarity
    2. Re-ranks candidates using combined semantic + lexical scoring
    
    Args:
        query: Search query text
        top_k: Number of results to return
        project_filter: Optional project name to filter results
        context_size: Number of prompts before/after to include (0 = no context)
    
    Returns:
        List of matching prompts with scores.
        If context_size > 0, each result includes 'context' with before/after prompts.
    """
    index, metadata, _ = load_index()
    
    if index is None or not metadata:
        print("Error: No index found. Run 'index' command first.")
        sys.exit(1)
    
    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Stage 1: Retrieve more candidates than needed for re-ranking
    # Fetch 50 or 5x top_k, whichever is larger
    candidate_k = max(50, top_k * 5)
    if project_filter:
        candidate_k *= 3  # Even more if filtering by project
    
    distances, indices = index.search(query_embedding, min(candidate_k, len(metadata)))
    
    # Build context index if needed
    context_idx = build_context_index(metadata) if context_size > 0 else None
    
    # Stage 2: Re-rank candidates with hybrid scoring
    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        prompt = metadata[idx].copy()
        semantic_score = float(dist)
        
        # Apply project filter
        if project_filter:
            if project_filter.lower() not in prompt["project"].lower():
                continue
        
        # Compute hybrid score (semantic + lexical)
        combined_score = compute_rerank_score(query, prompt["text"], semantic_score)
        
        prompt["semantic_score"] = semantic_score
        prompt["score"] = combined_score
        
        candidates.append(prompt)
    
    # Sort by combined score (descending)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top_k results
    results = candidates[:top_k]
    
    # Add context if requested
    if context_size > 0:
        for prompt in results:
            prompt["context"] = get_prompt_context(prompt, metadata, context_size, context_idx)
    
    return results


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_index(args):
    """Build or update the search index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    lock = FileLock(str(LOCK_PATH), timeout=LOCK_TIMEOUT)
    
    try:
        with lock:
            print("Acquired index lock.")
            
            index, metadata, workspace_states = incremental_index(force_full=args.force)
            
            if index is None:
                print("No index created.")
                return
            
            print("\nSaving index...")
            save_index(index, metadata, workspace_states)
            
            print(f"\nIndex saved to {INDEX_DIR}")
            print(f"  Total prompts indexed: {len(metadata)}")
            print(f"  Workspaces scanned: {len(workspace_states)}")
    
    except Timeout:
        print(f"Error: Could not acquire lock. Another indexing process may be running.")
        print(f"If you're sure no other process is running, delete {LOCK_PATH}")
        sys.exit(1)


def cmd_search(args):
    """Search for prompts."""
    query = " ".join(args.query)
    
    if not query.strip():
        print("Error: Please provide a search query.")
        sys.exit(1)
    
    context_size = getattr(args, 'context', 0) or 0
    
    results = search_prompts(
        query=query,
        top_k=args.top,
        project_filter=args.project,
        context_size=context_size,
    )
    
    if not results:
        print(f"No matches found for: \"{query}\"")
        return
    
    print(f"\nFound {len(results)} matches for: \"{query}\"\n")
    
    max_len = 150  # Truncate length for display
    
    for i, result in enumerate(results, 1):
        score = result["score"]
        project = result["project"]
        text = result["text"]
        
        # Format header
        print(f"[{i}] Score: {score:.2f} | Project: {project}")
        print("-" * 60)
        
        # Show context if available
        if context_size > 0 and "context" in result:
            ctx = result["context"]
            
            # Before context
            for j, before_prompt in enumerate(ctx["before"]):
                before_text = before_prompt["text"]
                display = before_text if len(before_text) <= max_len else before_text[:max_len] + "..."
                offset = -(len(ctx["before"]) - j)
                print(f"    [{offset:+d}] {display}")
            
            # The match (highlighted)
            display_text = text if len(text) <= max_len else text[:max_len] + "..."
            print(f"    [>>>] {display_text}")
            
            # After context
            for j, after_prompt in enumerate(ctx["after"], 1):
                after_text = after_prompt["text"]
                display = after_text if len(after_text) <= max_len else after_text[:max_len] + "..."
                print(f"    [+{j}] {display}")
        else:
            # No context - just show the match
            display_text = text if len(text) <= max_len else text[:max_len] + "..."
            print(f"    \"{display_text}\"")
        
        print()


def cmd_list_projects(args):
    """List all indexed projects."""
    _, metadata, workspace_states = load_index()
    
    if not metadata:
        print("No index found. Run 'index' command first.")
        return
    
    # Count prompts per project
    project_counts = {}
    for prompt in metadata:
        project = prompt["project"]
        project_counts[project] = project_counts.get(project, 0) + 1
    
    # Sort by prompt count
    sorted_projects = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nIndexed Projects ({len(sorted_projects)} total):\n")
    print(f"{'Project':<40} {'Prompts':>10}")
    print("-" * 52)
    
    for project, count in sorted_projects:
        print(f"{project:<40} {count:>10}")


def cmd_stats(args):
    """Show index statistics."""
    index, metadata, workspace_states = load_index()
    
    if index is None:
        print("No index found. Run 'index' command first.")
        return
    
    # Calculate stats
    total_prompts = len(metadata)
    total_workspaces = len(workspace_states)
    
    projects = set(p["project"] for p in metadata)
    
    # Get index file sizes
    faiss_size = FAISS_INDEX_PATH.stat().st_size if FAISS_INDEX_PATH.exists() else 0
    meta_size = METADATA_PATH.stat().st_size if METADATA_PATH.exists() else 0
    
    # Average prompt length
    avg_length = sum(len(p["text"]) for p in metadata) / total_prompts if total_prompts else 0
    
    print("\nCursor History Search - Index Statistics")
    print("=" * 45)
    print(f"Index location:      {INDEX_DIR}")
    print(f"Total prompts:       {total_prompts:,}")
    print(f"Unique projects:     {len(projects)}")
    print(f"Workspaces scanned:  {total_workspaces}")
    print(f"Avg prompt length:   {avg_length:.0f} chars")
    print(f"FAISS index size:    {faiss_size / 1024:.1f} KB")
    print(f"Metadata size:       {meta_size / 1024:.1f} KB")
    print(f"Embedding model:     {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")


def cmd_server(args):
    """Launch Streamlit web UI server."""
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("\nInstall it with:")
        print("  pip install streamlit")
        sys.exit(1)
    
    import subprocess
    import signal
    import atexit
    
    # Ensure index directory exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write the embedded Streamlit app to a temp file
    print(f"Writing Streamlit app to {STREAMLIT_APP_PATH}...")
    with open(STREAMLIT_APP_PATH, 'w') as f:
        f.write(STREAMLIT_APP_CODE)
    
    # Cleanup function
    def cleanup():
        if STREAMLIT_APP_PATH.exists():
            try:
                STREAMLIT_APP_PATH.unlink()
                print(f"\nCleaned up {STREAMLIT_APP_PATH}")
            except Exception:
                pass
    
    atexit.register(cleanup)
    
    # Build streamlit command
    port = getattr(args, 'port', 8501) or 8501
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(STREAMLIT_APP_PATH),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",  # Suppress torch.classes warnings
        "--logger.level", "error",  # Only show errors, not warnings
    ]
    
    print(f"\nStarting Streamlit server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        # Run streamlit
        process = subprocess.Popen(cmd)
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nShutting down...")
            process.terminate()
            cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()
    finally:
        cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Search across Cursor IDE chat history using semantic search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s index                          Build or update the search index
  %(prog)s search "bubble chart"          Search for prompts about bubble charts
  %(prog)s search "SQL query" --top 10    Get top 10 matches
  %(prog)s search "deploy" --project gnn  Filter by project name
  %(prog)s search "DDL" -C 2              Show 2 prompts before/after each match
  %(prog)s list-projects                  List all indexed projects
  %(prog)s stats                          Show index statistics
  %(prog)s --server                       Launch Streamlit web UI
  %(prog)s --server --port 8080           Launch on custom port
        """
    )
    
    # Top-level server arguments
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Launch Streamlit web UI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build or update the search index")
    index_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force full reindex (ignore cached state)"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for prompts")
    search_parser.add_argument(
        "query",
        nargs="+",
        help="Search query"
    )
    search_parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    search_parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Filter results by project name (partial match)"
    )
    search_parser.add_argument(
        "--context", "-C",
        type=int,
        default=0,
        metavar="N",
        help="Show N prompts before/after each match (like grep -C)"
    )
    
    # List projects command
    subparsers.add_parser("list-projects", help="List all indexed projects")
    
    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")
    
    args = parser.parse_args()
    
    # Handle --server flag (takes precedence)
    if args.server:
        cmd_server(args)
        return
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handler
    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "list-projects": cmd_list_projects,
        "stats": cmd_stats,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()

