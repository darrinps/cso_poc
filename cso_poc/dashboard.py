"""
Layer 2 — Executive Demo Dashboard  (Streamlit)

Left pane:   Guest chat interface
Right pane:  Three tabs:
  1. Decision Breadcrumbs  — real-time stream of policy decisions
  2. CSO Memory Vault      — Core / Tactical / Transient blocks with TTL
  3. Turn History          — envelope summaries per turn

Communicates with the CSO Orchestrator (Layer 4) via its HTTP API.

Architectural Decision: Three-tab layout mirrors three memory blocks
  The right-pane tabs directly correspond to the memory architecture:
  breadcrumbs show real-time decisions, the vault shows memory state
  with TTL decay visualization, and turn history shows the conversation
  arc.  This layout is designed for executive demonstrations where
  stakeholders need to see the reasoning process, not just the output.

Architectural Decision: Streamlit over custom frontend
  Streamlit provides rapid prototyping for data-heavy dashboards without
  requiring frontend engineering.  For a POC focused on demonstrating
  AI reasoning patterns, the tradeoff of less UI control for faster
  iteration is worthwhile.
"""

from __future__ import annotations

import math
import os

import requests
import streamlit as st

ORCHESTRATOR_URL = os.environ.get(
    "ORCHESTRATOR_URL", "http://cso-orchestrator:8001"
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CSO Executive Demo",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .breadcrumb-ok    { color: #22c55e; }
    .breadcrumb-warn  { color: #f59e0b; }
    .breadcrumb-err   { color: #ef4444; }
    .breadcrumb-mem   { color: #8b5cf6; }
    .breadcrumb-decay { color: #6b7280; text-decoration: line-through; }
    .ttl-bar { height: 8px; border-radius: 4px; margin: 2px 0 6px 0; }
    .ttl-high   { background: #22c55e; }
    .ttl-medium { background: #f59e0b; }
    .ttl-low    { background: #ef4444; }
    .ttl-dead   { background: #374151; }
    .fact-core      { border-left: 3px solid #3b82f6; padding-left: 8px; }
    .fact-tactical  { border-left: 3px solid #f59e0b; padding-left: 8px; }
    .fact-transient { border-left: 3px solid #8b5cf6; padding-left: 8px; }
    .fact-scrubbed  { border-left: 3px solid #6b7280; padding-left: 8px;
                      opacity: 0.5; text-decoration: line-through; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "breadcrumb_cursor" not in st.session_state:
    st.session_state.breadcrumb_cursor = 0
if "breadcrumbs" not in st.session_state:
    st.session_state.breadcrumbs = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def send_message(text: str) -> str | None:
    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/chat",
            json={"message": text}, timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response received.")
    except requests.exceptions.ConnectionError:
        return "[Connection error — is the orchestrator running?]"
    except Exception as exc:
        return f"[Error: {exc}]"


def fetch_breadcrumbs() -> list[dict]:
    try:
        resp = requests.get(
            f"{ORCHESTRATOR_URL}/breadcrumbs",
            params={"since": st.session_state.breadcrumb_cursor},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.breadcrumb_cursor = data.get(
            "total", st.session_state.breadcrumb_cursor
        )
        return data.get("breadcrumbs", [])
    except Exception:
        return []


def fetch_memory_vault() -> dict:
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}/memory-vault", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {
            "core_block": [], "recall_block": [], "transient_block": [],
            "scrub_log": [], "stats": {},
        }


def fetch_memory() -> dict:
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}/memory", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"turns": [], "turn_count": 0}


# ---------------------------------------------------------------------------
# TTL rendering helpers
# ---------------------------------------------------------------------------

def ttl_bar_html(ttl_seconds: float | None, max_seconds: float = 172800) -> str:
    """Render a coloured TTL progress bar."""
    if ttl_seconds is None:
        return '<div class="ttl-bar ttl-high" style="width:100%"></div>'
    if ttl_seconds <= 0:
        return '<div class="ttl-bar ttl-dead" style="width:100%"></div>'
    pct = min(ttl_seconds / max_seconds * 100, 100)
    if pct > 50:
        css = "ttl-high"
    elif pct > 15:
        css = "ttl-medium"
    else:
        css = "ttl-low"
    return f'<div class="ttl-bar {css}" style="width:{pct:.1f}%"></div>'


def format_ttl(ttl_seconds: float | None) -> str:
    if ttl_seconds is None:
        return "Permanent"
    if ttl_seconds <= 0:
        return "EXPIRED"
    hours = int(ttl_seconds // 3600)
    minutes = int((ttl_seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {int(ttl_seconds % 60)}s"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown("## Cognitive Singularity Orchestrator -- Executive Demo")
st.markdown(
    "*Centralized reasoning with deterministic execution. "
    "No agent-to-agent communication.*"
)
st.divider()

col_chat, col_feed = st.columns([1, 1], gap="large")

# ---------------------------------------------------------------------------
# Left pane — Guest Chat
# ---------------------------------------------------------------------------

with col_chat:
    st.markdown("### Guest Chat")

    btn_cols = st.columns(4)
    with btn_cols[0]:
        if st.button("Complex request", use_container_width=True,
                      disabled=st.session_state.processing):
            st.session_state.pending_message = (
                "I'd like a late checkout at 5 PM, a bottle of Macallan 12 "
                "sent to my room, and please apply my Suite Night Award for tonight."
            )
    with btn_cols[1]:
        if st.button("WV follow-up", use_container_width=True,
                      disabled=st.session_state.processing
                      or len(st.session_state.messages) == 0):
            st.session_state.pending_message = (
                "Wait, did you handle my WV stay too?"
            )
    with btn_cols[2]:
        if st.button("WV persistence", use_container_width=True,
                      disabled=st.session_state.processing
                      or len(st.session_state.messages) < 2):
            st.session_state.pending_message = (
                "Can you confirm my WV arrival is still on file?"
            )
    with btn_cols[3]:
        if st.button("Flight Delay Recovery", use_container_width=True,
                      disabled=st.session_state.processing):
            st.session_state.pending_message = (
                "[SSE EVENT] Flight delay detected for G-3003: "
                "+3 hours. Trigger high-value guest recovery protocol."
            )

    chat_container = st.container(height=460)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "guest":
                with st.chat_message("user"):
                    st.markdown(msg["text"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["text"])

    user_input = st.chat_input(
        "Type a guest request...",
        disabled=st.session_state.processing,
    )

    message_to_send = None
    if user_input:
        message_to_send = user_input
    elif "pending_message" in st.session_state:
        message_to_send = st.session_state.pop("pending_message")

    if message_to_send:
        st.session_state.messages.append({"role": "guest", "text": message_to_send})
        st.session_state.processing = True
        with st.spinner("CSO reasoning..."):
            response = send_message(message_to_send)
        st.session_state.messages.append({"role": "cso", "text": response})
        st.session_state.processing = False
        new_crumbs = fetch_breadcrumbs()
        st.session_state.breadcrumbs.extend(new_crumbs)
        st.rerun()


# ---------------------------------------------------------------------------
# Right pane — Breadcrumbs / Memory Vault / Turn History
# ---------------------------------------------------------------------------

with col_feed:
    tab_crumbs, tab_vault, tab_history = st.tabs([
        "Decision Breadcrumbs", "CSO Memory Vault", "Turn History",
    ])

    # ── Tab 1: Breadcrumbs ──────────────────────────────────────────────
    with tab_crumbs:
        st.markdown("### Live Breadcrumb Feed")
        if st.button("Refresh", key="refresh_crumbs"):
            new_crumbs = fetch_breadcrumbs()
            st.session_state.breadcrumbs.extend(new_crumbs)

        feed_container = st.container(height=460)
        with feed_container:
            if not st.session_state.breadcrumbs:
                st.markdown("*No breadcrumbs yet. Send a guest message to start.*")
            else:
                for crumb in st.session_state.breadcrumbs:
                    result = crumb.get("result", "")
                    policy = crumb.get("policy_reference", "")
                    trace = crumb.get("trace_id", "")[:8]

                    if "SCRUBBED" in result or "FORGOTTEN" in result:
                        css = "breadcrumb-decay"
                        icon = "~~"
                    elif "ESCALAT" in result or "DENIED" in result:
                        css = "breadcrumb-warn"
                        icon = "!!"
                    elif "ERROR" in result:
                        css = "breadcrumb-err"
                        icon = "XX"
                    elif "MEMORY" in policy:
                        css = "breadcrumb-mem"
                        icon = "??"
                    else:
                        css = "breadcrumb-ok"
                        icon = "OK"

                    st.markdown(
                        f'<span class="{css}">`[{icon}]`</span> '
                        f"`{trace}` | **{policy}**  \n"
                        f'<small>`{crumb.get("action_taken", "")}`</small>  \n'
                        f'<small class="{css}">{result}</small>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

    # ── Tab 2: Memory Vault ─────────────────────────────────────────────
    with tab_vault:
        st.markdown("### CSO Memory Vault")
        if st.button("Refresh", key="refresh_vault"):
            pass

        vault = fetch_memory_vault()
        stats = vault.get("stats", {})

        mcols = st.columns(4)
        mcols[0].metric("Core (permanent)", stats.get("core_count", 0))
        mcols[1].metric("Tactical (48h TTL)", stats.get("tactical_count", 0))
        mcols[2].metric("Transient (live)", stats.get("transient_count", 0))
        mcols[3].metric("Total scrubbed", stats.get("total_scrubbed", 0))

        vault_container = st.container(height=380)
        with vault_container:
            # Core block
            st.markdown("#### Core Block (Permanent)")
            core = vault.get("core_block", [])
            if core:
                for f in core:
                    st.markdown(
                        f'<div class="fact-core">'
                        f'<strong>{f["domain"]}</strong>: {f["fact"]}'
                        f'<br/><small>TTL: Permanent</small>'
                        f'{ttl_bar_html(None)}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*Empty — no core facts stored yet.*")

            st.markdown("")

            # Tactical block
            st.markdown("#### Tactical Block (48h TTL)")
            tactical = vault.get("recall_block", [])
            if tactical:
                for f in tactical:
                    ttl = f.get("ttl_seconds")
                    ttl_str = format_ttl(ttl)
                    trace = f.get("trace_id", "")[:8]
                    st.markdown(
                        f'<div class="fact-tactical">'
                        f'<strong>{f["domain"]}</strong>: {f["fact"]}'
                        f'<br/><small>TTL: {ttl_str} | Trace: {trace} | '
                        f'Tags: {", ".join(f.get("tags", []))}</small>'
                        f'{ttl_bar_html(ttl)}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*Empty — no tactical facts stored yet.*")

            st.markdown("")

            # Transient block
            st.markdown("#### Transient Block (Sentiment / State)")
            transient = vault.get("transient_block", [])
            if transient:
                for f in transient:
                    ttl = f.get("ttl_seconds")
                    ttl_str = format_ttl(ttl)
                    st.markdown(
                        f'<div class="fact-transient">'
                        f'<strong>{f["domain"]}</strong>: {f["fact"]}'
                        f'<br/><small>TTL: {ttl_str}</small>'
                        f'{ttl_bar_html(ttl, max_seconds=600)}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '*Empty — transient facts have been decayed '
                    '(sentiment cleared on intent resolution).*'
                )

            # Scrub log
            scrub_log = vault.get("scrub_log", [])
            if scrub_log:
                st.markdown("")
                st.markdown("#### Decay Audit Log")
                for entry in reversed(scrub_log[-10:]):
                    reason = entry.get("reason", "ttl_expired")
                    st.markdown(
                        f'<div class="fact-scrubbed">'
                        f'<strong>[{entry["tier"]}]</strong> {entry["fact"]}'
                        f'<br/><small>Scrubbed: {entry["scrubbed_at"]} | '
                        f'Reason: {reason}</small></div>',
                        unsafe_allow_html=True,
                    )

    # ── Tab 3: Turn History ─────────────────────────────────────────────
    with tab_history:
        st.markdown("### Turn History")
        if st.button("Refresh", key="refresh_history"):
            pass

        mem = fetch_memory()
        turn_count = mem.get("turn_count", 0)
        st.metric("Turns stored", turn_count)

        for turn in mem.get("turns", []):
            with st.expander(
                f"Turn {turn['turn']} — {turn['status']} | "
                f"{turn['trace_id'][:8]}..."
            ):
                st.markdown(f"**Objective:** {turn['objective']}")

                if turn.get("domain_assertions"):
                    st.markdown("**Domain Assertions:**")
                    for a in turn["domain_assertions"]:
                        st.markdown(f"- {a}")

                if turn.get("compromises"):
                    st.markdown("**Compromises:**")
                    for c in turn["compromises"]:
                        st.markdown(f"- `{c['action']}`: {c['rationale']}")

                if turn.get("escalations"):
                    st.markdown("**Escalations:**")
                    for e in turn["escalations"]:
                        st.markdown(f"- {e}")

                if turn.get("actions_taken"):
                    st.markdown("**Actions Executed:**")
                    for a in turn["actions_taken"]:
                        tag = " (compromise)" if a.get("is_compromise") else ""
                        st.markdown(f"- `{a['tool']}`{tag}")
