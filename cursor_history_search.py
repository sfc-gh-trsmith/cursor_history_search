#!/usr/bin/env python3
"""
Cursor History Search & Analytics Tool

A CLI tool to search across all Cursor IDE chat history using semantic search,
with comprehensive analytics from conversations, daily stats, and plans.
Extracts data from Cursor's SQLite databases, generates embeddings with
sentence-transformers, and indexes with FAISS for fast similarity search.

Usage:
    python cursor_history_search.py index           # Build/update the index and analytics DB
    python cursor_history_search.py search "query"  # Search for prompts
    python cursor_history_search.py analytics       # Show productivity analytics
    python cursor_history_search.py timeline        # Show activity timeline
    python cursor_history_search.py patterns        # Extract common prompt patterns
    python cursor_history_search.py export          # Export analytics to markdown/JSON
    python cursor_history_search.py list-projects   # List indexed projects
    python cursor_history_search.py stats           # Show index statistics
    python cursor_history_search.py --server        # Launch Streamlit web UI
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

# Global storage path (for daily stats)
GLOBAL_STORAGE_PATH = CURSOR_BASE_PATH / "globalStorage"
GLOBAL_STATE_DB = GLOBAL_STORAGE_PATH / "state.vscdb"

# Plans directory
PLANS_DIR = Path.home() / ".cursor" / "plans"

# Index storage location
INDEX_DIR = Path.home() / ".cursor_history_index"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
WORKSPACE_STATE_PATH = INDEX_DIR / "workspace_state.json"
ANALYTICS_DB_PATH = INDEX_DIR / "analytics.db"
LOCK_PATH = INDEX_DIR / "index.lock"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Lock timeout in seconds
LOCK_TIMEOUT = 10

# Streamlit app temp file
STREAMLIT_APP_PATH = INDEX_DIR / "_streamlit_app.py"


# =============================================================================
# Analytics Database Schema
# =============================================================================

ANALYTICS_SCHEMA = """
-- Core prompts table with per-prompt metrics
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    workspace_hash TEXT NOT NULL,
    project TEXT NOT NULL,
    prompt_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    timestamp_ms INTEGER,
    generation_uuid TEXT,
    conversation_id TEXT,
    UNIQUE(workspace_hash, prompt_index)
);

-- Conversation-level metrics from composerData
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    workspace_hash TEXT NOT NULL,
    project TEXT NOT NULL,
    name TEXT,
    created_at_ms INTEGER,
    last_updated_ms INTEGER,
    total_lines_added INTEGER DEFAULT 0,
    total_lines_removed INTEGER DEFAULT 0,
    files_changed_count INTEGER DEFAULT 0,
    files_changed_list TEXT,
    context_usage_pct REAL,
    prompt_count INTEGER DEFAULT 0
);

-- Daily aggregate stats from aiCodeTracking
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    tab_suggested_lines INTEGER DEFAULT 0,
    tab_accepted_lines INTEGER DEFAULT 0,
    composer_suggested_lines INTEGER DEFAULT 0,
    composer_accepted_lines INTEGER DEFAULT 0,
    acceptance_rate_tab REAL,
    acceptance_rate_composer REAL
);

-- Plans from ~/.cursor/plans/
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    name TEXT,
    file_path TEXT,
    created_at_ms INTEGER,
    last_updated_ms INTEGER,
    todo_count INTEGER DEFAULT 0,
    completed_count INTEGER DEFAULT 0,
    workspace_hash TEXT
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_prompts_project ON prompts(project);
CREATE INDEX IF NOT EXISTS idx_prompts_workspace ON prompts(workspace_hash);
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project);
CREATE INDEX IF NOT EXISTS idx_conversations_last_updated ON conversations(last_updated_ms);
"""

ANALYTICS_VIEWS = """
-- Aggregate view for project summary
DROP VIEW IF EXISTS project_summary;
CREATE VIEW project_summary AS
SELECT 
    c.project,
    COUNT(DISTINCT c.id) as conversation_count,
    (SELECT COUNT(*) FROM prompts p WHERE p.project = c.project) as prompt_count,
    COALESCE(SUM(c.total_lines_added), 0) as total_lines_added,
    COALESCE(SUM(c.total_lines_removed), 0) as total_lines_removed,
    COALESCE(SUM(c.files_changed_count), 0) as total_files_changed,
    MAX(c.last_updated_ms) as last_activity_ms
FROM conversations c
GROUP BY c.project;

-- Weekly activity view
DROP VIEW IF EXISTS weekly_activity;
CREATE VIEW weekly_activity AS
SELECT 
    strftime('%Y-W%W', date) as week,
    SUM(composer_suggested_lines) as suggested,
    SUM(composer_accepted_lines) as accepted,
    ROUND(
        CASE WHEN SUM(composer_suggested_lines) > 0 
        THEN (SUM(composer_accepted_lines) * 100.0 / SUM(composer_suggested_lines))
        ELSE 0 END, 1
    ) as avg_acceptance_pct
FROM daily_stats
GROUP BY week
ORDER BY week DESC;
"""


def init_analytics_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the analytics SQLite database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript(ANALYTICS_SCHEMA)
    
    # Create views
    cursor.executescript(ANALYTICS_VIEWS)
    
    conn.commit()
    return conn


def get_analytics_db(db_path: Path = None) -> sqlite3.Connection:
    """Get a connection to the analytics database."""
    if db_path is None:
        db_path = ANALYTICS_DB_PATH
    
    if not db_path.exists():
        return init_analytics_db(db_path)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# Embedded Streamlit App Code (Native Python for introspection)
# =============================================================================

def _streamlit_app():
    """
    Complete Streamlit app code.
    
    This function contains the entire Streamlit UI and is extracted via
    introspection for deployment. All imports are inside the function
    to make extraction self-contained.
    """
    # All imports inside function for self-contained extraction
    import json
    import sqlite3
    import subprocess
    import sys
    from datetime import datetime
    from pathlib import Path

    import streamlit as st

    # Configuration - must match main script
    INDEX_DIR = Path.home() / ".cursor_history_index"
    FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
    METADATA_PATH = INDEX_DIR / "metadata.json"
    WORKSPACE_STATE_PATH = INDEX_DIR / "workspace_state.json"
    ANALYTICS_DB_PATH = INDEX_DIR / "analytics.db"
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

    # =========================================================================
    # Analytics Database Functions
    # =========================================================================

    def get_analytics_db():
        """Get connection to analytics database."""
        if not ANALYTICS_DB_PATH.exists():
            return None
        conn = sqlite3.connect(str(ANALYTICS_DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn

    def get_dashboard_kpis():
        """Get key performance indicators for dashboard."""
        conn = get_analytics_db()
        if not conn:
            return None
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM prompts")
        total_prompts = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*), COALESCE(SUM(total_lines_added), 0), COALESCE(SUM(total_lines_removed), 0)
            FROM conversations
        """)
        row = cursor.fetchone()
        total_conversations, lines_added, lines_removed = row[0], row[1], row[2]
        
        cursor.execute("""
            SELECT SUM(composer_suggested_lines), SUM(composer_accepted_lines), COUNT(*)
            FROM daily_stats
        """)
        row = cursor.fetchone()
        suggested, accepted, days = row[0] or 0, row[1] or 0, row[2]
        acceptance_rate = (accepted / suggested * 100) if suggested > 0 else 0
        
        cursor.execute("SELECT COUNT(DISTINCT project) FROM prompts")
        unique_projects = cursor.fetchone()[0]
        
        conn.close()
        return {
            "total_prompts": total_prompts,
            "total_conversations": total_conversations,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "acceptance_rate": acceptance_rate,
            "days_tracked": days,
            "unique_projects": unique_projects,
        }

    def get_daily_trends(days=30):
        """Get daily stats for charts."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?", (days,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(results))

    def get_project_summary():
        """Get per-project aggregates."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM project_summary ORDER BY total_lines_added DESC")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_recent_conversations(limit=10):
        """Get recent conversations."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM conversations 
            WHERE last_updated_ms IS NOT NULL
            ORDER BY last_updated_ms DESC LIMIT ?
        """, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_top_conversations_by_impact(limit=10):
        """Get top conversations by code impact."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *, (total_lines_added + total_lines_removed) as total_impact
            FROM conversations ORDER BY total_impact DESC LIMIT ?
        """, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_timeline_data(project=None, limit=50):
        """Get timeline data with optional filtering."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        query = "SELECT * FROM conversations WHERE last_updated_ms IS NOT NULL"
        params = []
        if project:
            query += " AND project LIKE ?"
            params.append(f"%{project}%")
        query += " ORDER BY last_updated_ms DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_weekly_activity():
        """Get weekly aggregated stats."""
        conn = get_analytics_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM weekly_activity")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    # =========================================================================
    # Search Functions
    # =========================================================================

    @st.cache_resource
    def get_model():
        return SentenceTransformer(EMBEDDING_MODEL)

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
        ctx_idx = {}
        for p in metadata:
            key = (p["workspace_hash"], p["prompt_index"])
            ctx_idx[key] = p
        return ctx_idx

    def get_session_prompts(workspace_hash, metadata):
        session_prompts = [p for p in metadata if p["workspace_hash"] == workspace_hash]
        return sorted(session_prompts, key=lambda x: x["prompt_index"])

    def get_prompt_context(result, context_idx, context_size=2):
        ws_hash = result["workspace_hash"]
        prompt_idx = result["prompt_index"]
        before, after = [], []
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
        query_lower = query.lower()
        text_lower = text.lower()
        exact_match_bonus = 0.5 if query_lower in text_lower else 0.0
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        word_overlap_score = len(query_words & text_words) / len(query_words) if query_words else 0.0
        return 0.5 * semantic_score + 0.3 * exact_match_bonus + 0.2 * word_overlap_score

    def search_prompts(query, top_k=10, project_filter=None, context_size=0):
        index, metadata, _ = load_index()
        if index is None or not metadata:
            return []
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        candidate_k = max(50, top_k * 5)
        if project_filter and project_filter != "All":
            candidate_k *= 3
        distances, indices = index.search(query_embedding, min(candidate_k, len(metadata)))
        context_idx = build_context_index(metadata) if context_size > 0 else None
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            prompt = metadata[idx].copy()
            semantic_score = float(dist)
            if project_filter and project_filter != "All":
                if project_filter.lower() not in prompt["project"].lower():
                    continue
            combined_score = compute_rerank_score(query, prompt["text"], semantic_score)
            prompt["semantic_score"] = semantic_score
            prompt["score"] = combined_score
            candidates.append(prompt)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:top_k]
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
        # MAIN_SCRIPT_PATH is injected by cmd_server() when generating the app
        script_path = Path(MAIN_SCRIPT_PATH) if 'MAIN_SCRIPT_PATH' in globals() else None
        if not script_path or not script_path.exists():
            script_path = Path(__file__).parent / "cursor_history_search.py"
        if not script_path.exists():
            script_path = INDEX_DIR.parent / "cursor_history_search.py"
        if not script_path.exists():
            return False, f"Could not find cursor_history_search.py script"
        result = subprocess.run(
            [sys.executable, str(script_path), "index", "--force"],
            capture_output=True, text=True,
        )
        return result.returncode == 0, result.stdout + result.stderr

    # =========================================================================
    # Page Config and CSS
    # =========================================================================

    st.set_page_config(
        page_title="Cursor History Analytics",
        page_icon="📊",
        layout="wide",
    )

    st.markdown("""
    <style>
        .kpi-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 10px;
        }
        .kpi-value { font-size: 2em; font-weight: bold; }
        .kpi-label { font-size: 0.9em; opacity: 0.9; }
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
        }
        .timeline-card {
            background-color: rgba(100, 100, 100, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .context-label { font-size: 0.75em; color: #aaa; font-weight: bold; margin-right: 8px; }
        .prompt-index {
            font-size: 0.8em;
            color: #ccc;
            background-color: rgba(100, 100, 100, 0.3);
            padding: 2px 6px;
            border-radius: 3px;
            margin-right: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    # Sidebar
    # =========================================================================

    with st.sidebar:
        st.title("📊 Cursor Analytics")
        st.markdown("---")
        
        if st.button("🔄 Re-index", use_container_width=True):
            with st.spinner("Re-indexing..."):
                success, output = run_reindex()
                if success:
                    st.success("Complete!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Failed")
                    st.code(output)

        st.markdown("---")
        
        kpis = get_dashboard_kpis()
        if kpis:
            st.metric("Prompts", f"{kpis['total_prompts']:,}")
            st.metric("Conversations", f"{kpis['total_conversations']:,}")
            st.metric("Acceptance Rate", f"{kpis['acceptance_rate']:.1f}%")
            st.metric("Lines Added", f"{kpis['lines_added']:,}")

    # =========================================================================
    # Main Content - Tab Navigation
    # =========================================================================

    if st.session_state.view_session:
        # Session View Mode
        session_info = st.session_state.view_session
        ws_hash = session_info["workspace_hash"]
        project = session_info["project"]
        highlight_idx = st.session_state.highlight_prompt_idx
        
        if st.button("← Back"):
            st.session_state.view_session = None
            st.session_state.highlight_prompt_idx = None
            st.rerun()
        
        st.title(f"Session: `{project}`")
        _, metadata, _ = load_index()
        session_prompts = get_session_prompts(ws_hash, metadata)
        st.info(f"**{len(session_prompts)} prompts** in this session")
        
        for prompt in session_prompts:
            idx = prompt["prompt_index"]
            is_highlight = (idx == highlight_idx)
            css_class = "session-prompt-highlight" if is_highlight else "session-prompt"
            label = ">>> MATCH" if is_highlight else f"#{idx}"
            st.markdown(f'<div class="{css_class}"><span class="prompt-index">{label}</span>{prompt["text"]}</div>', unsafe_allow_html=True)

    else:
        # Tab Navigation
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard", "🔍 Search", "📈 Analytics", "📅 Timeline", "🔤 Patterns", "📥 Export"])
        
        # =====================================================================
        # Dashboard Tab
        # =====================================================================
        with tab1:
            st.title("Dashboard")
            
            kpis = get_dashboard_kpis()
            if kpis:
                # KPI Cards
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["total_prompts"]:,}</div><div class="kpi-label">Prompts</div></div>', unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["total_conversations"]:,}</div><div class="kpi-label">Conversations</div></div>', unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["lines_added"]:,}</div><div class="kpi-label">Lines Added</div></div>', unsafe_allow_html=True)
                with cols[3]:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["acceptance_rate"]:.1f}%</div><div class="kpi-label">Acceptance Rate</div></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Daily Trend Chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Daily Activity")
                    daily = get_daily_trends(30)
                    if daily:
                        import pandas as pd
                        df = pd.DataFrame(daily)
                        if not df.empty and "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            st.line_chart(df.set_index("date")[["composer_suggested_lines", "composer_accepted_lines"]])
                
                with col2:
                    st.subheader("Top Projects")
                    projects = get_project_summary()[:10]
                    if projects:
                        import pandas as pd
                        df = pd.DataFrame(projects)
                        if not df.empty:
                            st.bar_chart(df.set_index("project")["total_lines_added"])
                
                # Recent Conversations
                st.subheader("Recent Conversations")
                recent = get_recent_conversations(5)
                for conv in recent:
                    name = conv.get("name") or "(unnamed)"
                    ts = conv.get("last_updated_ms")
                    date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
                    lines = conv.get("total_lines_added", 0) + conv.get("total_lines_removed", 0)
                    st.markdown(f'<div class="timeline-card"><b>{name}</b><br/><small>{conv["project"]} | {date_str} | {lines} lines</small></div>', unsafe_allow_html=True)
            else:
                st.warning("No analytics data. Run 'Re-index' to build the database.")
        
        # =====================================================================
        # Search Tab
        # =====================================================================
        with tab2:
            st.title("Semantic Search")
            
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            with col1:
                query = st.text_input("Search", placeholder="Enter your search query...", label_visibility="collapsed")
            with col2:
                stats = get_stats()
                project_options = ["All"] + (list(stats["projects"].keys()) if stats else [])
                project_filter = st.selectbox("Project", project_options, label_visibility="collapsed")
            with col3:
                top_k = st.selectbox("Results", [5, 10, 20, 50], index=1, label_visibility="collapsed")
            with col4:
                context_size = st.selectbox("Context", [0, 1, 2, 3, 5], index=0, format_func=lambda x: f"±{x}" if x > 0 else "None", label_visibility="collapsed")
            
            if query:
                with st.spinner("Searching..."):
                    results = search_prompts(query, top_k=top_k, project_filter=project_filter if project_filter != "All" else None, context_size=context_size)
                
                if results:
                    st.success(f"Found {len(results)} matches")
                    for i, result in enumerate(results, 1):
                        col_h, col_b = st.columns([5, 1])
                        with col_h:
                            st.markdown(f"**[{i}] Score: {result['score']:.2f}** — `{result['project']}`")
                        with col_b:
                            if st.button("View", key=f"s_{i}"):
                                st.session_state.view_session = {"workspace_hash": result["workspace_hash"], "project": result["project"]}
                                st.session_state.highlight_prompt_idx = result["prompt_index"]
                                st.rerun()
                        
                        if context_size > 0 and "context" in result:
                            ctx = result["context"]
                            for j, bp in enumerate(ctx["before"]):
                                text = bp["text"][:200] + "..." if len(bp["text"]) > 200 else bp["text"]
                                st.markdown(f'<div class="context-prompt">{text}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="match-prompt">{result["text"]}</div>', unsafe_allow_html=True)
                            for j, ap in enumerate(ctx["after"]):
                                text = ap["text"][:200] + "..." if len(ap["text"]) > 200 else ap["text"]
                                st.markdown(f'<div class="context-prompt">{text}</div>', unsafe_allow_html=True)
                        else:
                            st.code(result["text"], language=None)
                        st.markdown("---")
                else:
                    st.warning("No matches found")
        
        # =====================================================================
        # Analytics Tab
        # =====================================================================
        with tab3:
            st.title("Detailed Analytics")
            
            kpis = get_dashboard_kpis()
            if kpis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weekly Trends")
                    weekly = get_weekly_activity()
                    if weekly:
                        import pandas as pd
                        df = pd.DataFrame(weekly)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True)
                
                with col2:
                    st.subheader("Top Conversations by Impact")
                    top = get_top_conversations_by_impact(10)
                    for c in top:
                        name = c.get("name") or "(unnamed)"
                        impact = c.get("total_impact", 0)
                        st.markdown(f"**{name[:40]}** — {c['project']} — {impact:,} lines")
                
                st.subheader("Project Breakdown")
                projects = get_project_summary()
                if projects:
                    import pandas as pd
                    df = pd.DataFrame(projects)
                    st.dataframe(df, use_container_width=True)
        
        # =====================================================================
        # Timeline Tab
        # =====================================================================
        with tab4:
            st.title("Activity Timeline")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                project_filter_tl = st.text_input("Filter by project", placeholder="Project name...")
            with col2:
                limit_tl = st.selectbox("Show", [20, 50, 100], index=0)
            
            conversations = get_timeline_data(project=project_filter_tl if project_filter_tl else None, limit=limit_tl)
            
            for conv in conversations:
                ts = conv.get("last_updated_ms")
                date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
                name = conv.get("name") or "(unnamed)"
                lines_added = conv.get("total_lines_added", 0)
                lines_removed = conv.get("total_lines_removed", 0)
                files_count = conv.get("files_changed_count", 0)
                
                st.markdown(f"""
                <div class="timeline-card">
                    <b>{name}</b><br/>
                    <small>📁 {conv['project']} | 📅 {date_str}</small><br/>
                    <small>+{lines_added} / -{lines_removed} lines | {files_count} files</small>
                </div>
                """, unsafe_allow_html=True)
        
        # =====================================================================
        # Patterns Tab
        # =====================================================================
        with tab5:
            st.title("Prompt Patterns")
            
            _, metadata, _ = load_index()
            if metadata:
                # Pattern analysis
                patterns = {}
                for prompt in metadata:
                    text = prompt["text"].strip().lower()
                    words = text.split()[:5]
                    if len(words) >= 2:
                        pattern_key = " ".join(words[:3])
                        if pattern_key not in patterns:
                            patterns[pattern_key] = []
                        patterns[pattern_key].append(prompt)
                
                sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
                
                st.subheader("Common Prompt Patterns")
                for pattern, prompts in sorted_patterns[:20]:
                    if len(prompts) >= 2:
                        st.markdown(f"**{pattern}...** — {len(prompts)} occurrences")
                
                # Keyword frequency
                st.subheader("Top Keywords")
                word_freq = {}
                stopwords = {"the", "a", "an", "is", "it", "to", "and", "of", "in", "for", "on", "with", "this", "that", "i", "you", "we", "be", "are", "was", "have", "has", "do", "does", "can", "will", "would", "should", "could", "if", "then", "else", "when", "what", "how", "why", "where", "which", "who", "or", "not", "no", "yes", "my", "your", "our", "their", "its", "as", "at", "by", "from", "into", "about", "all", "any", "but", "so", "up", "out", "just", "now", "only", "also", "than", "more", "some", "very", "too", "each", "other", "such", "make", "like", "use", "get", "add", "new"}
                for prompt in metadata:
                    words = prompt["text"].lower().split()
                    for word in words:
                        word = "".join(c for c in word if c.isalnum())
                        if len(word) > 3 and word not in stopwords:
                            word_freq[word] = word_freq.get(word, 0) + 1
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
                
                import pandas as pd
                df = pd.DataFrame(sorted_words, columns=["Keyword", "Count"])
                st.bar_chart(df.set_index("Keyword"))
            else:
                st.warning("No prompts found. Run 'Re-index' first.")
        
        # =====================================================================
        # Export Tab
        # =====================================================================
        with tab6:
            st.title("Export Data")
            
            format_opt = st.selectbox("Format", ["Markdown", "JSON"])
            
            if st.button("Generate Report"):
                kpis = get_dashboard_kpis()
                projects = get_project_summary()
                weekly = get_weekly_activity()
                top = get_top_conversations_by_impact(10)
                
                if format_opt == "JSON":
                    data = {
                        "generated_at": datetime.now().isoformat(),
                        "kpis": kpis,
                        "projects": projects,
                        "weekly": weekly,
                        "top_impact": top,
                    }
                    output = json.dumps(data, indent=2, default=str)
                    st.code(output, language="json")
                    st.download_button("Download JSON", output, "cursor_analytics.json", "application/json")
                else:
                    lines = [
                        "# Cursor History Analytics Report",
                        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
                        "\n## Overview",
                        f"- Prompts: {kpis['total_prompts']:,}" if kpis else "",
                        f"- Conversations: {kpis['total_conversations']:,}" if kpis else "",
                        f"- Lines Added: {kpis['lines_added']:,}" if kpis else "",
                        f"- Acceptance Rate: {kpis['acceptance_rate']:.1f}%" if kpis else "",
                    ]
                    output = "\n".join(lines)
                    st.markdown(output)
                    st.download_button("Download Markdown", output, "cursor_analytics.md", "text/markdown")


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


def extract_generations_from_workspace(workspace_path: Path) -> List[Dict[str, Any]]:
    """
    Extract generation data from a workspace's state.vscdb SQLite database.
    Returns list of generation dictionaries with UUIDs and timestamps.
    """
    db_path = workspace_path / "state.vscdb"
    if not db_path.exists():
        return []
    
    generations = []
    project_name = get_project_name_from_workspace(workspace_path) or workspace_path.name
    
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            ("aiService.generations",)
        )
        row = cursor.fetchone()
        
        if row and row[0]:
            try:
                gen_data = json.loads(row[0])
                if isinstance(gen_data, list):
                    for item in gen_data:
                        if isinstance(item, dict):
                            generations.append({
                                "generation_uuid": item.get("generationUUID"),
                                "timestamp_ms": item.get("unixMs"),
                                "type": item.get("type"),
                                "text_description": item.get("textDescription", ""),
                                "project": project_name,
                                "workspace_hash": workspace_path.name,
                            })
            except json.JSONDecodeError:
                pass
        
        conn.close()
    except sqlite3.Error:
        pass
    
    return generations


def extract_composer_data_from_workspace(workspace_path: Path) -> List[Dict[str, Any]]:
    """
    Extract composer/conversation data from a workspace's state.vscdb.
    Returns list of conversation dictionaries with metrics.
    """
    db_path = workspace_path / "state.vscdb"
    if not db_path.exists():
        return []
    
    conversations = []
    project_name = get_project_name_from_workspace(workspace_path) or workspace_path.name
    
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            ("composer.composerData",)
        )
        row = cursor.fetchone()
        
        if row and row[0]:
            try:
                composer_data = json.loads(row[0])
                all_composers = composer_data.get("allComposers", [])
                
                for item in all_composers:
                    if isinstance(item, dict) and item.get("type") == "head":
                        # Parse files from subtitle (comma-separated list)
                        subtitle = item.get("subtitle", "")
                        files_list = [f.strip() for f in subtitle.split(",") if f.strip()] if subtitle else []
                        
                        conversations.append({
                            "id": item.get("composerId"),
                            "workspace_hash": workspace_path.name,
                            "project": project_name,
                            "name": item.get("name"),
                            "created_at_ms": item.get("createdAt"),
                            "last_updated_ms": item.get("lastUpdatedAt"),
                            "total_lines_added": item.get("totalLinesAdded", 0),
                            "total_lines_removed": item.get("totalLinesRemoved", 0),
                            "files_changed_count": item.get("filesChangedCount", 0),
                            "files_changed_list": json.dumps(files_list),
                            "context_usage_pct": item.get("contextUsagePercent"),
                        })
            except json.JSONDecodeError:
                pass
        
        conn.close()
    except sqlite3.Error:
        pass
    
    return conversations


def extract_daily_stats() -> List[Dict[str, Any]]:
    """
    Extract daily code tracking stats from global storage.
    Returns list of daily stats dictionaries.
    """
    if not GLOBAL_STATE_DB.exists():
        return []
    
    daily_stats = []
    
    try:
        uri = f"file:{GLOBAL_STATE_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        # Find all aiCodeTracking.dailyStats keys
        cursor.execute(
            "SELECT key, value FROM ItemTable WHERE key LIKE 'aiCodeTracking.dailyStats.%'"
        )
        
        for row in cursor.fetchall():
            key = row[0]
            value = row[1]
            
            if value:
                try:
                    data = json.loads(value)
                    date = data.get("date")
                    if date:
                        tab_suggested = data.get("tabSuggestedLines", 0)
                        tab_accepted = data.get("tabAcceptedLines", 0)
                        composer_suggested = data.get("composerSuggestedLines", 0)
                        composer_accepted = data.get("composerAcceptedLines", 0)
                        
                        # Calculate acceptance rates
                        tab_rate = (tab_accepted / tab_suggested) if tab_suggested > 0 else None
                        composer_rate = (composer_accepted / composer_suggested) if composer_suggested > 0 else None
                        
                        daily_stats.append({
                            "date": date,
                            "tab_suggested_lines": tab_suggested,
                            "tab_accepted_lines": tab_accepted,
                            "composer_suggested_lines": composer_suggested,
                            "composer_accepted_lines": composer_accepted,
                            "acceptance_rate_tab": tab_rate,
                            "acceptance_rate_composer": composer_rate,
                        })
                except json.JSONDecodeError:
                    pass
        
        conn.close()
    except sqlite3.Error:
        pass
    
    return daily_stats


def extract_plans() -> List[Dict[str, Any]]:
    """
    Extract plan data from ~/.cursor/plans/ directory.
    Returns list of plan dictionaries with metadata.
    """
    if not PLANS_DIR.exists():
        return []
    
    plans = []
    
    for plan_file in PLANS_DIR.glob("*.plan.md"):
        try:
            content = plan_file.read_text()
            
            # Parse YAML frontmatter
            plan_data = {
                "id": plan_file.stem,
                "name": None,
                "file_path": str(plan_file),
                "created_at_ms": int(plan_file.stat().st_ctime * 1000),
                "last_updated_ms": int(plan_file.stat().st_mtime * 1000),
                "todo_count": 0,
                "completed_count": 0,
                "workspace_hash": None,
            }
            
            # Try to extract YAML frontmatter
            if content.startswith("---"):
                try:
                    end_idx = content.index("---", 3)
                    frontmatter = content[3:end_idx].strip()
                    
                    # Simple YAML parsing for name and todos
                    for line in frontmatter.split("\n"):
                        if line.startswith("name:"):
                            plan_data["name"] = line.split(":", 1)[1].strip()
                        elif "status: completed" in line:
                            plan_data["completed_count"] += 1
                        elif "status:" in line and "- id:" in frontmatter:
                            plan_data["todo_count"] += 1
                    
                    # Count todos more accurately
                    plan_data["todo_count"] = content.count("- id:")
                    plan_data["completed_count"] = content.count("status: completed")
                    
                except ValueError:
                    pass
            
            plans.append(plan_data)
            
        except Exception:
            pass
    
    return plans


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


def extract_all_analytics_data() -> Dict[str, Any]:
    """
    Extract all analytics data from Cursor storage.
    
    Returns:
        Dict with keys: prompts, generations, conversations, daily_stats, plans
    """
    if not WORKSPACE_STORAGE_PATH.exists():
        print(f"Error: Cursor workspace storage not found at {WORKSPACE_STORAGE_PATH}")
        sys.exit(1)
    
    all_prompts = []
    all_generations = []
    all_conversations = []
    workspace_states = {}
    
    workspace_dirs = [d for d in WORKSPACE_STORAGE_PATH.iterdir() if d.is_dir()]
    print(f"Scanning {len(workspace_dirs)} workspaces for analytics data...")
    
    for workspace_path in workspace_dirs:
        db_path = workspace_path / "state.vscdb"
        if not db_path.exists():
            continue
        
        mtime = db_path.stat().st_mtime
        
        # Extract all data types from this workspace
        prompts = extract_prompts_from_workspace(workspace_path)
        generations = extract_generations_from_workspace(workspace_path)
        conversations = extract_composer_data_from_workspace(workspace_path)
        
        if prompts:
            project_name = prompts[0]["project"]
            print(f"  {project_name}: {len(prompts)} prompts, {len(conversations)} conversations")
            all_prompts.extend(prompts)
        
        all_generations.extend(generations)
        all_conversations.extend(conversations)
        
        workspace_states[workspace_path.name] = {
            "mtime": mtime,
            "prompt_count": len(prompts),
            "project": prompts[0]["project"] if prompts else None,
        }
    
    # Extract global data
    print("Extracting daily stats from global storage...")
    daily_stats = extract_daily_stats()
    print(f"  Found {len(daily_stats)} days of stats")
    
    print("Extracting plans...")
    plans = extract_plans()
    print(f"  Found {len(plans)} plans")
    
    return {
        "prompts": all_prompts,
        "generations": all_generations,
        "conversations": all_conversations,
        "daily_stats": daily_stats,
        "plans": plans,
        "workspace_states": workspace_states,
    }


# =============================================================================
# Analytics Database Population
# =============================================================================

def populate_analytics_db(analytics_data: Dict[str, Any], db_path: Path = None) -> None:
    """
    Populate the analytics database with extracted data.
    
    Args:
        analytics_data: Dict from extract_all_analytics_data()
        db_path: Path to analytics.db (defaults to ANALYTICS_DB_PATH)
    """
    if db_path is None:
        db_path = ANALYTICS_DB_PATH
    
    conn = init_analytics_db(db_path)
    cursor = conn.cursor()
    
    # Clear existing data for full refresh
    cursor.execute("DELETE FROM prompts")
    cursor.execute("DELETE FROM conversations")
    cursor.execute("DELETE FROM daily_stats")
    cursor.execute("DELETE FROM plans")
    
    # Insert prompts
    prompts = analytics_data.get("prompts", [])
    generations = analytics_data.get("generations", [])
    
    # Build generation lookup by text for matching
    gen_lookup = {}
    for gen in generations:
        if gen.get("text_description"):
            # Use first 100 chars as key
            key = gen["text_description"][:100].lower()
            gen_lookup[key] = gen
    
    for prompt in prompts:
        # Try to find matching generation
        gen_uuid = None
        timestamp_ms = None
        text_key = prompt["text"][:100].lower()
        if text_key in gen_lookup:
            gen = gen_lookup[text_key]
            gen_uuid = gen.get("generation_uuid")
            timestamp_ms = gen.get("timestamp_ms")
        
        cursor.execute("""
            INSERT OR REPLACE INTO prompts 
            (workspace_hash, project, prompt_index, text, timestamp_ms, generation_uuid)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prompt["workspace_hash"],
            prompt["project"],
            prompt["prompt_index"],
            prompt["text"],
            timestamp_ms,
            gen_uuid,
        ))
    
    # Insert conversations
    conversations = analytics_data.get("conversations", [])
    for conv in conversations:
        cursor.execute("""
            INSERT OR REPLACE INTO conversations 
            (id, workspace_hash, project, name, created_at_ms, last_updated_ms,
             total_lines_added, total_lines_removed, files_changed_count, 
             files_changed_list, context_usage_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conv["id"],
            conv["workspace_hash"],
            conv["project"],
            conv.get("name"),
            conv.get("created_at_ms"),
            conv.get("last_updated_ms"),
            conv.get("total_lines_added", 0),
            conv.get("total_lines_removed", 0),
            conv.get("files_changed_count", 0),
            conv.get("files_changed_list"),
            conv.get("context_usage_pct"),
        ))
    
    # Insert daily stats
    daily_stats = analytics_data.get("daily_stats", [])
    for stat in daily_stats:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_stats 
            (date, tab_suggested_lines, tab_accepted_lines, 
             composer_suggested_lines, composer_accepted_lines,
             acceptance_rate_tab, acceptance_rate_composer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            stat["date"],
            stat.get("tab_suggested_lines", 0),
            stat.get("tab_accepted_lines", 0),
            stat.get("composer_suggested_lines", 0),
            stat.get("composer_accepted_lines", 0),
            stat.get("acceptance_rate_tab"),
            stat.get("acceptance_rate_composer"),
        ))
    
    # Insert plans
    plans = analytics_data.get("plans", [])
    for plan in plans:
        cursor.execute("""
            INSERT OR REPLACE INTO plans 
            (id, name, file_path, created_at_ms, last_updated_ms, 
             todo_count, completed_count, workspace_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            plan["id"],
            plan.get("name"),
            plan.get("file_path"),
            plan.get("created_at_ms"),
            plan.get("last_updated_ms"),
            plan.get("todo_count", 0),
            plan.get("completed_count", 0),
            plan.get("workspace_hash"),
        ))
    
    conn.commit()
    conn.close()
    
    print(f"Analytics DB populated: {len(prompts)} prompts, {len(conversations)} conversations, "
          f"{len(daily_stats)} daily stats, {len(plans)} plans")


# =============================================================================
# Analytics Query Functions
# =============================================================================

def get_dashboard_kpis(db_path: Path = None) -> Dict[str, Any]:
    """
    Return key performance indicators for the dashboard.
    
    Returns:
        Dict with: total_prompts, total_conversations, total_lines_added,
                   total_lines_removed, acceptance_rate, days_tracked
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    # Get prompt count
    cursor.execute("SELECT COUNT(*) FROM prompts")
    total_prompts = cursor.fetchone()[0]
    
    # Get conversation stats
    cursor.execute("""
        SELECT COUNT(*), 
               COALESCE(SUM(total_lines_added), 0),
               COALESCE(SUM(total_lines_removed), 0)
        FROM conversations
    """)
    row = cursor.fetchone()
    total_conversations = row[0]
    total_lines_added = row[1]
    total_lines_removed = row[2]
    
    # Get acceptance rate
    cursor.execute("""
        SELECT SUM(composer_suggested_lines), SUM(composer_accepted_lines), COUNT(*)
        FROM daily_stats
    """)
    row = cursor.fetchone()
    suggested = row[0] or 0
    accepted = row[1] or 0
    days_tracked = row[2]
    acceptance_rate = (accepted / suggested * 100) if suggested > 0 else 0
    
    # Get unique projects
    cursor.execute("SELECT COUNT(DISTINCT project) FROM prompts")
    unique_projects = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_prompts": total_prompts,
        "total_conversations": total_conversations,
        "total_lines_added": total_lines_added,
        "total_lines_removed": total_lines_removed,
        "acceptance_rate": acceptance_rate,
        "days_tracked": days_tracked,
        "unique_projects": unique_projects,
    }


def get_daily_trends(db_path: Path = None, days: int = 30) -> List[Dict[str, Any]]:
    """
    Return daily stats for trend charts.
    
    Args:
        db_path: Path to analytics.db
        days: Number of days to return (0 for all)
    
    Returns:
        List of dicts with daily stats, ordered by date
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    if days > 0:
        cursor.execute("""
            SELECT * FROM daily_stats 
            ORDER BY date DESC 
            LIMIT ?
        """, (days,))
    else:
        cursor.execute("SELECT * FROM daily_stats ORDER BY date DESC")
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Return in chronological order
    return list(reversed(results))


def get_project_summary(db_path: Path = None) -> List[Dict[str, Any]]:
    """
    Return per-project aggregates.
    
    Returns:
        List of dicts with project stats, ordered by activity
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM project_summary 
        ORDER BY total_lines_added DESC
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_recent_conversations(db_path: Path = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return recent conversations with metrics.
    
    Returns:
        List of conversation dicts, ordered by last_updated
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM conversations 
        WHERE last_updated_ms IS NOT NULL
        ORDER BY last_updated_ms DESC 
        LIMIT ?
    """, (limit,))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_top_conversations_by_impact(db_path: Path = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return conversations ranked by code impact (lines added + removed).
    
    Returns:
        List of conversation dicts
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT *, (total_lines_added + total_lines_removed) as total_impact
        FROM conversations 
        ORDER BY total_impact DESC 
        LIMIT ?
    """, (limit,))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_weekly_activity(db_path: Path = None) -> List[Dict[str, Any]]:
    """
    Return weekly aggregated activity.
    
    Returns:
        List of weekly stats
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM weekly_activity")
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_timeline_data(
    db_path: Path = None, 
    project: str = None, 
    start_date: str = None,
    end_date: str = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Return conversation timeline data with optional filtering.
    
    Args:
        db_path: Path to analytics.db
        project: Filter by project name (partial match)
        start_date: Filter conversations after this date (YYYY-MM-DD)
        end_date: Filter conversations before this date (YYYY-MM-DD)
        limit: Maximum number of results
    
    Returns:
        List of conversation dicts with timeline data
    """
    conn = get_analytics_db(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM conversations WHERE 1=1"
    params = []
    
    if project:
        query += " AND project LIKE ?"
        params.append(f"%{project}%")
    
    if start_date:
        # Convert date to milliseconds
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        query += " AND last_updated_ms >= ?"
        params.append(start_ms)
    
    if end_date:
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        query += " AND last_updated_ms <= ?"
        params.append(end_ms)
    
    query += " ORDER BY last_updated_ms DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


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
    """Build or update the search index and analytics database."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    lock = FileLock(str(LOCK_PATH), timeout=LOCK_TIMEOUT)
    
    try:
        with lock:
            print("Acquired index lock.")
            
            # Extract all analytics data (includes prompts)
            print("\n--- Extracting Analytics Data ---")
            analytics_data = extract_all_analytics_data()
            
            prompts = analytics_data["prompts"]
            workspace_states = analytics_data["workspace_states"]
            
            if not prompts:
                print("No prompts found to index.")
                return
            
            # Build FAISS index
            print(f"\n--- Building FAISS Index ---")
            print(f"Generating embeddings for {len(prompts)} prompts...")
            texts = [p["text"] for p in prompts]
            embeddings = generate_embeddings(texts)
            
            print("Building FAISS index...")
            index = create_faiss_index(embeddings)
            
            print("\nSaving FAISS index...")
            save_index(index, prompts, workspace_states)
            
            # Populate analytics database
            print("\n--- Populating Analytics Database ---")
            populate_analytics_db(analytics_data)
            
            print(f"\n--- Index Complete ---")
            print(f"Location: {INDEX_DIR}")
            print(f"  FAISS index: {len(prompts)} prompts")
            print(f"  Analytics DB: {ANALYTICS_DB_PATH.name}")
            print(f"  Workspaces: {len(workspace_states)}")
    
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


def cmd_analytics(args):
    """Show productivity analytics from the SQLite database."""
    if not ANALYTICS_DB_PATH.exists():
        print("No analytics database found. Run 'index' command first.")
        return
    
    # Get KPIs
    kpis = get_dashboard_kpis()
    
    print("\n" + "=" * 60)
    print("CURSOR HISTORY ANALYTICS")
    print("=" * 60)
    
    # Overview
    print("\n📊 OVERVIEW")
    print("-" * 40)
    print(f"  Total Prompts:        {kpis['total_prompts']:,}")
    print(f"  Total Conversations:  {kpis['total_conversations']:,}")
    print(f"  Unique Projects:      {kpis['unique_projects']}")
    print(f"  Days Tracked:         {kpis['days_tracked']}")
    
    # Code Impact
    print("\n💻 CODE IMPACT")
    print("-" * 40)
    print(f"  Lines Added:          {kpis['total_lines_added']:,}")
    print(f"  Lines Removed:        {kpis['total_lines_removed']:,}")
    print(f"  Net Change:           {kpis['total_lines_added'] - kpis['total_lines_removed']:+,}")
    print(f"  Acceptance Rate:      {kpis['acceptance_rate']:.1f}%")
    
    # Top Projects
    print("\n📁 TOP PROJECTS BY CODE IMPACT")
    print("-" * 40)
    projects = get_project_summary()[:10]
    if projects:
        print(f"  {'Project':<30} {'Lines+':>8} {'Lines-':>8} {'Convs':>6}")
        print("  " + "-" * 54)
        for p in projects:
            print(f"  {p['project'][:30]:<30} {p['total_lines_added']:>8,} "
                  f"{p['total_lines_removed']:>8,} {p['conversation_count']:>6}")
    
    # Recent Activity
    print("\n🕐 RECENT CONVERSATIONS")
    print("-" * 40)
    recent = get_recent_conversations(limit=5)
    for conv in recent:
        name = conv.get('name') or '(unnamed)'
        name = name[:40] + '...' if len(name) > 40 else name
        lines = conv.get('total_lines_added', 0) + conv.get('total_lines_removed', 0)
        ts = conv.get('last_updated_ms')
        if ts:
            date_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
        else:
            date_str = 'Unknown'
        print(f"  [{date_str}] {name} ({lines} lines)")
    
    # Weekly Trend
    print("\n📈 WEEKLY TREND")
    print("-" * 40)
    weekly = get_weekly_activity()[:4]
    if weekly:
        print(f"  {'Week':<12} {'Suggested':>12} {'Accepted':>12} {'Rate':>8}")
        print("  " + "-" * 46)
        for w in weekly:
            rate = f"{w['avg_acceptance_pct']:.1f}%" if w.get('avg_acceptance_pct') else "N/A"
            print(f"  {w['week']:<12} {w['suggested']:>12,} {w['accepted']:>12,} {rate:>8}")
    
    print()


def cmd_timeline(args):
    """Show activity timeline with optional filtering."""
    if not ANALYTICS_DB_PATH.exists():
        print("No analytics database found. Run 'index' command first.")
        return
    
    project = getattr(args, 'project', None)
    start_date = getattr(args, 'start', None)
    end_date = getattr(args, 'end', None)
    limit = getattr(args, 'limit', 20) or 20
    
    conversations = get_timeline_data(
        project=project,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    if not conversations:
        print("No conversations found matching criteria.")
        return
    
    print("\n" + "=" * 70)
    print("CONVERSATION TIMELINE")
    if project:
        print(f"Filtered by project: {project}")
    if start_date or end_date:
        print(f"Date range: {start_date or 'start'} to {end_date or 'now'}")
    print("=" * 70 + "\n")
    
    for conv in conversations:
        ts = conv.get('last_updated_ms')
        if ts:
            date_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
        else:
            date_str = 'Unknown date'
        
        name = conv.get('name') or '(unnamed conversation)'
        project_name = conv.get('project', 'Unknown')
        lines_added = conv.get('total_lines_added', 0)
        lines_removed = conv.get('total_lines_removed', 0)
        files_count = conv.get('files_changed_count', 0)
        
        # Parse files list
        files_list = []
        if conv.get('files_changed_list'):
            try:
                files_list = json.loads(conv['files_changed_list'])
            except:
                pass
        
        print(f"📅 {date_str}")
        print(f"   📁 {project_name}")
        print(f"   💬 {name}")
        print(f"   📊 +{lines_added} / -{lines_removed} lines | {files_count} files")
        if files_list:
            print(f"   📄 {', '.join(files_list[:5])}" + ("..." if len(files_list) > 5 else ""))
        print()


def cmd_patterns(args):
    """Extract common prompt patterns using clustering."""
    if not ANALYTICS_DB_PATH.exists():
        print("No analytics database found. Run 'index' command first.")
        return
    
    # Load prompts
    _, metadata, _ = load_index()
    if not metadata:
        print("No prompts found. Run 'index' command first.")
        return
    
    print("\n" + "=" * 60)
    print("PROMPT PATTERN ANALYSIS")
    print("=" * 60)
    
    # Simple pattern analysis without external dependencies
    # Group by common starting phrases
    patterns = {}
    for prompt in metadata:
        text = prompt['text'].strip().lower()
        # Get first few words as pattern key
        words = text.split()[:5]
        if len(words) >= 2:
            pattern_key = ' '.join(words[:3])
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(prompt)
    
    # Sort by frequency
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Show top patterns
    print("\n📊 COMMON PROMPT PATTERNS")
    print("-" * 50)
    print(f"{'Pattern Start':<40} {'Count':>8}")
    print("-" * 50)
    
    for pattern, prompts in sorted_patterns[:20]:
        if len(prompts) >= 2:  # Only show patterns with 2+ occurrences
            print(f"{pattern[:40]:<40} {len(prompts):>8}")
    
    # Keyword frequency
    print("\n🔤 TOP KEYWORDS")
    print("-" * 50)
    
    word_freq = {}
    stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'in', 'for', 'on', 'with', 'this', 'that', 'i', 'you', 'we', 'be', 'are', 'was', 'have', 'has', 'do', 'does', 'can', 'will', 'would', 'should', 'could', 'if', 'then', 'else', 'when', 'what', 'how', 'why', 'where', 'which', 'who', 'or', 'not', 'no', 'yes', 'my', 'your', 'our', 'their', 'its', 'as', 'at', 'by', 'from', 'into', 'about', 'all', 'any', 'but', 'so', 'up', 'out', 'just', 'now', 'only', 'also', 'than', 'more', 'some', 'very', 'too', 'each', 'other', 'such', 'make', 'like', 'use', 'get', 'add', 'new'}
    
    for prompt in metadata:
        words = prompt['text'].lower().split()
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Keyword':<20} {'Count':>8}")
    print("-" * 30)
    for word, count in sorted_words[:20]:
        print(f"{word:<20} {count:>8}")
    
    print()


def cmd_export(args):
    """Export analytics data to various formats."""
    if not ANALYTICS_DB_PATH.exists():
        print("No analytics database found. Run 'index' command first.")
        return
    
    output_format = getattr(args, 'format', 'markdown') or 'markdown'
    output_file = getattr(args, 'output', None)
    
    # Gather data
    kpis = get_dashboard_kpis()
    projects = get_project_summary()
    recent = get_recent_conversations(limit=20)
    weekly = get_weekly_activity()
    top_impact = get_top_conversations_by_impact(limit=10)
    
    if output_format == 'json':
        data = {
            "generated_at": datetime.now().isoformat(),
            "kpis": kpis,
            "projects": projects,
            "recent_conversations": recent,
            "weekly_activity": weekly,
            "top_impact_conversations": top_impact,
        }
        output = json.dumps(data, indent=2, default=str)
        
    else:  # markdown
        lines = [
            "# Cursor History Analytics Report",
            f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## Overview\n",
            f"- **Total Prompts:** {kpis['total_prompts']:,}",
            f"- **Total Conversations:** {kpis['total_conversations']:,}",
            f"- **Unique Projects:** {kpis['unique_projects']}",
            f"- **Days Tracked:** {kpis['days_tracked']}",
            f"- **Lines Added:** {kpis['total_lines_added']:,}",
            f"- **Lines Removed:** {kpis['total_lines_removed']:,}",
            f"- **Acceptance Rate:** {kpis['acceptance_rate']:.1f}%",
            "\n## Projects by Code Impact\n",
            "| Project | Lines Added | Lines Removed | Conversations |",
            "|---------|-------------|---------------|---------------|",
        ]
        
        for p in projects[:15]:
            lines.append(f"| {p['project']} | {p['total_lines_added']:,} | {p['total_lines_removed']:,} | {p['conversation_count']} |")
        
        lines.extend([
            "\n## Weekly Activity\n",
            "| Week | Suggested | Accepted | Rate |",
            "|------|-----------|----------|------|",
        ])
        
        for w in weekly[:8]:
            rate = f"{w['avg_acceptance_pct']:.1f}%" if w.get('avg_acceptance_pct') else "N/A"
            lines.append(f"| {w['week']} | {w['suggested']:,} | {w['accepted']:,} | {rate} |")
        
        lines.extend([
            "\n## Top Conversations by Impact\n",
            "| Conversation | Project | Lines Changed |",
            "|--------------|---------|---------------|",
        ])
        
        for c in top_impact:
            name = c.get('name') or '(unnamed)'
            name = name[:40] + '...' if len(name) > 40 else name
            impact = c.get('total_lines_added', 0) + c.get('total_lines_removed', 0)
            lines.append(f"| {name} | {c['project']} | {impact:,} |")
        
        output = '\n'.join(lines)
    
    if output_file:
        Path(output_file).write_text(output)
        print(f"Report exported to: {output_file}")
    else:
        print(output)


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
    
    import inspect
    import subprocess
    import signal
    import atexit
    import textwrap
    
    # Ensure index directory exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract the Streamlit app code via introspection
    print(f"Extracting Streamlit app via introspection...")
    source = inspect.getsource(_streamlit_app)
    
    # Parse to get just the function body (remove def line and docstring, then dedent)
    lines = source.split('\n')
    
    # Skip 'def _streamlit_app():' line and find where docstring ends
    body_start = 1  # Start after def line
    in_docstring = False
    docstring_delimiter = None
    
    for i, line in enumerate(lines[1:], 1):  # Skip def line
        stripped = line.strip()
        
        if not in_docstring:
            # Check if this line starts a docstring
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_delimiter = stripped[:3]
                in_docstring = True
                # Check if docstring ends on same line
                if stripped.count(docstring_delimiter) >= 2:
                    body_start = i + 1
                    break
            elif stripped and not stripped.startswith('#'):
                # First non-empty, non-comment line that's not a docstring
                body_start = i
                break
        else:
            # We're inside a docstring, look for the end
            if docstring_delimiter in stripped:
                body_start = i + 1
                break
    
    # Extract body and dedent
    body_lines = lines[body_start:]
    body = textwrap.dedent('\n'.join(body_lines))
    
    # Inject the main script path so run_reindex() can find it
    # Get the path to this script file
    this_script = Path(__file__).resolve()
    script_path_injection = f'MAIN_SCRIPT_PATH = "{this_script}"\n\n'
    body = script_path_injection + body
    
    # Write the extracted code to temp file
    print(f"Writing Streamlit app to {STREAMLIT_APP_PATH}...")
    with open(STREAMLIT_APP_PATH, 'w') as f:
        f.write(body)
    
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
    
    # Analytics command
    subparsers.add_parser("analytics", help="Show productivity analytics from the database")
    
    # Timeline command
    timeline_parser = subparsers.add_parser("timeline", help="Show activity timeline")
    timeline_parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Filter by project name (partial match)"
    )
    timeline_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    timeline_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    timeline_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum results (default: 20)"
    )
    
    # Patterns command
    subparsers.add_parser("patterns", help="Extract common prompt patterns")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export analytics data")
    export_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (prints to stdout if not specified)"
    )
    
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
        "analytics": cmd_analytics,
        "timeline": cmd_timeline,
        "patterns": cmd_patterns,
        "export": cmd_export,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()

