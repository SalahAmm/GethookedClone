from __future__ import annotations

import collections

import streamlit as st

CSS = """
<style>
:root {
  --bg: #0E1117; --surface: #1A1D24; --border: #2A2E38;
  --accent: #FF3D5A; --teal: #22D3EE; --text: #E6E9EF; --muted: #9AA3B2;
}
[data-testid="stSidebar"] { background: var(--surface); }
.block-container { padding-top: 2rem; }
.kpi-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.25rem; height: 100%;
}
.kpi-card .kpi-value {
  font-family: 'Fira Code', monospace; font-size: 2rem;
  font-weight: 700; color: var(--text);
}
.kpi-card .kpi-label {
  color: var(--muted); font-size: 0.85rem;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.kpi-card .kpi-value.hot { color: var(--accent); }
.pill {
  display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
  font-family: 'Fira Code', monospace; font-weight: 600; font-size: 0.9rem;
  color: #0E1117;
}
.pill.hot { background: #FF3D5A; }
.pill.warm { background: #F59E0B; }
.pill.weak { background: #9AA3B2; }
[data-testid="stExpander"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px;
}
hr { border-color: var(--border); }
</style>
"""


def build_metrics(rows: list[dict]) -> dict:
    scores = [r["hook_score"] for r in rows if r.get("hook_score")]
    types = [r["hook_type"] for r in rows if r.get("hook_type")]
    return {
        "total": len(rows),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "top_hook": collections.Counter(types).most_common(1)[0][0] if types else "—",
    }


def score_pill(score: int) -> str:
    band = "hot" if score >= 8 else "warm" if score >= 6 else "weak"
    return f'<span class="pill {band}">{score}/10</span>'


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)