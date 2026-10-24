from __future__ import annotations

import html

import streamlit as st

from site_context import get_active_stock


WORKFLOW_STEPS = [
    ("DETAIL", "🔬", "銘柄詳細", "pages/16_Stock_Detail.py"),
    ("ME", "🚀", "ME", "pages/14_Multiple_Expansion_Hunter.py"),
    ("ENTRY", "🎯", "Entry", "pages/8_Short_Cover_Hunter.py"),
    ("PRETRADE", "🛡️", "Pre-Trade", "pages/9_Pre_Trade_Check.py"),
    ("RECORD", "📚", "記録", "pages/📖_レポートビューア.py"),
]


V3_GLOBAL_CSS = """
<style>
:root {
    --v3-blue: #2563eb;
    --v3-blue-soft: #eff6ff;
    --v3-slate-900: #0f172a;
    --v3-slate-700: #334155;
    --v3-slate-500: #64748b;
    --v3-border: rgba(148, 163, 184, 0.22);
    --v3-panel: rgba(255, 255, 255, 0.92);
    --v3-panel-soft: rgba(248, 250, 252, 0.88);
    --v3-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
}

/* Page width and rhythm */
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1500px;
    padding-top: 1.05rem;
    padding-bottom: 3.5rem;
}

main h1, main h2, main h3, main h4 {
    letter-spacing: -0.015em;
}

main h2 {
    margin-top: 1.45rem;
    margin-bottom: 0.65rem;
}

main h3 {
    margin-top: 1.15rem;
    margin-bottom: 0.55rem;
}

/* KPI / metric cards */
main [data-testid="stMetric"] {
    min-height: 92px;
    padding: 0.82rem 0.95rem;
    border: 1px solid var(--v3-border);
    border-radius: 14px;
    background:
        linear-gradient(180deg, rgba(255,255,255,.98) 0%, rgba(248,250,252,.92) 100%);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
}

main [data-testid="stMetricLabel"] {
    color: var(--v3-slate-500);
    font-weight: 700;
}

main [data-testid="stMetricValue"] {
    color: var(--v3-slate-900);
    font-weight: 820;
    letter-spacing: -0.02em;
}

/* Bordered Streamlit containers */
main [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 15px;
    border-color: var(--v3-border);
    background: var(--v3-panel);
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
}

/* Buttons */
main .stButton > button {
    min-height: 2.65rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.34);
    font-weight: 750;
    transition:
        transform 120ms ease,
        box-shadow 120ms ease,
        border-color 120ms ease;
}

main .stButton > button:hover:not(:disabled) {
    transform: translateY(-1px);
    border-color: rgba(37, 99, 235, 0.42);
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.10);
}

main .stButton > button[kind="primary"] {
    border: none;
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    color: #ffffff;
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.18);
}

main .stButton > button:disabled {
    opacity: 0.58;
}

/* Page-link buttons in the main area */
main [data-testid="stPageLink"] a {
    min-height: 2.6rem;
    padding: 0.55rem 0.72rem;
    border: 1px solid var(--v3-border);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.88);
    font-weight: 720;
    transition:
        transform 120ms ease,
        border-color 120ms ease,
        box-shadow 120ms ease;
}

main [data-testid="stPageLink"] a:hover {
    transform: translateY(-1px);
    border-color: rgba(37, 99, 235, 0.38);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08);
}

/* Tabs */
main [data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.28rem;
    padding: 0.28rem;
    border-radius: 12px;
    background: rgba(241, 245, 249, 0.86);
}

main [data-testid="stTabs"] [data-baseweb="tab"] {
    min-height: 2.55rem;
    padding: 0.55rem 0.82rem;
    border-radius: 9px;
    font-weight: 720;
}

main [data-testid="stTabs"] [aria-selected="true"] {
    background: #ffffff;
    color: #1d4ed8;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
}

/* Expanders */
main [data-testid="stExpander"] {
    border-color: var(--v3-border);
    border-radius: 13px;
    overflow: hidden;
    background: rgba(255,255,255,.76);
}

main [data-testid="stExpander"] summary {
    min-height: 2.7rem;
    font-weight: 720;
}

/* Inputs */
main div[data-baseweb="input"] > div,
main div[data-baseweb="textarea"] > div,
main div[data-baseweb="select"] > div {
    border-radius: 10px;
}

main textarea {
    border-radius: 10px !important;
}

/* Dataframes / tables */
main [data-testid="stDataFrame"] {
    border: 1px solid var(--v3-border);
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
main [data-testid="stAlert"] {
    border-radius: 12px;
}

/* Reduce visual noise between tightly stacked blocks */
main [data-testid="stVerticalBlock"] > div {
    scroll-margin-top: 5rem;
}

.v3-section-shell {
    margin: 1.15rem 0 0.7rem 0;
    padding: 0.78rem 0.92rem;
    border-left: 4px solid var(--v3-blue);
    border-radius: 0 12px 12px 0;
    background: linear-gradient(90deg, rgba(239,246,255,.82), rgba(255,255,255,.35));
}

.v3-section-title {
    color: var(--v3-slate-900);
    font-size: 1rem;
    font-weight: 820;
}

.v3-section-caption {
    margin-top: 0.12rem;
    color: var(--v3-slate-500);
    font-size: 0.76rem;
    line-height: 1.5;
}

.v3-inline-badge {
    display: inline-block;
    padding: .22rem .48rem;
    border-radius: 999px;
    background: var(--v3-blue-soft);
    color: #1d4ed8;
    font-size: .69rem;
    font-weight: 800;
}
</style>
"""


def render_v3_design_system() -> None:
    """Apply shared visual tokens and interaction styling across the V3 site."""
    st.markdown(V3_GLOBAL_CSS, unsafe_allow_html=True)


def render_v3_section(
    title: str,
    caption: str = "",
    *,
    icon: str = "",
) -> None:
    icon_text = f"{icon} " if icon else ""
    caption_html = (
        f'<div class="v3-section-caption">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    st.markdown(
        (
            '<div class="v3-section-shell">'
            f'<div class="v3-section-title">{icon_text}{html.escape(title)}</div>'
            f'{caption_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_v3_page_header(
    *,
    title: str,
    subtitle: str,
    stage: str,
    icon: str,
    badge: str | None = None,
    show_workflow: bool = True,
) -> None:
    ticker, name, source = get_active_stock()
    stock_text = "未選択"
    if ticker:
        stock_text = f"{ticker}｜{name}" if name else ticker

    badge_html = (
        f'<span class="v3-page-badge">{html.escape(str(badge))}</span>'
        if badge
        else ""
    )

    st.markdown(
        f"""
        <style>
        .v3-page-shell {{
            margin: 0.15rem 0 1rem 0;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(148,163,184,.22);
            border-radius: 16px;
            background:
                radial-gradient(circle at 90% 0%, rgba(59,130,246,.10), transparent 22rem),
                linear-gradient(135deg,#ffffff 0%,#f8fbff 100%);
        }}
        .v3-page-kicker {{
            font-size: .69rem;
            font-weight: 800;
            letter-spacing: .11em;
            color: #2563eb;
        }}
        .v3-page-title {{
            margin-top: .15rem;
            font-size: 1.45rem;
            font-weight: 850;
            color: #0f172a;
            letter-spacing: -.02em;
        }}
        .v3-page-subtitle {{
            margin-top: .28rem;
            color: #64748b;
            font-size: .84rem;
            line-height: 1.6;
        }}
        .v3-page-meta {{
            margin-top: .55rem;
            display: flex;
            flex-wrap: wrap;
            gap: .45rem;
            align-items: center;
            font-size: .74rem;
            color: #475569;
        }}
        .v3-page-stock {{
            padding: .28rem .55rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-weight: 750;
        }}
        .v3-page-badge {{
            padding: .28rem .55rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            font-weight: 750;
        }}
        .v3-workflow {{
            margin: .15rem 0 1rem 0;
            padding: .55rem .7rem;
            border-radius: 13px;
            border: 1px solid rgba(148,163,184,.18);
            background: rgba(248,250,252,.75);
        }}
        </style>

        <div class="v3-page-shell">
          <div class="v3-page-kicker">TAKASAN STOCK ANALYZER V3</div>
          <div class="v3-page-title">{icon} {html.escape(title)}</div>
          <div class="v3-page-subtitle">{html.escape(subtitle)}</div>
          <div class="v3-page-meta">
            <span class="v3-page-stock">現在の銘柄: {html.escape(stock_text)}</span>
            {badge_html}
            <span>共通銘柄はページ間で引き継がれます</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source and source not in {"sidebar", stage}:
        st.caption(f"↳ {source} から引き継ぎ")

    if not show_workflow:
        return

    st.markdown('<div class="v3-workflow">', unsafe_allow_html=True)
    cols = st.columns(len(WORKFLOW_STEPS))
    current_stage = str(stage or "").upper()
    for col, (step, step_icon, label, path) in zip(cols, WORKFLOW_STEPS):
        with col:
            is_current = step == current_stage
            st.page_link(
                path,
                label=(f"● {label}" if is_current else label),
                icon=step_icon,
                use_container_width=True,
                disabled=is_current,
            )
    st.markdown("</div>", unsafe_allow_html=True)
