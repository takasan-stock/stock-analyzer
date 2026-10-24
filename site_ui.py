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
