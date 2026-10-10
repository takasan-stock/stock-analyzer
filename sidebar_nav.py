from __future__ import annotations

import streamlit as st


NAV_SECTIONS = [
    {
        "label": "今日の売買フロー",
        "tone": "blue",
        "pages": [
            ("dashboard_app.py", "ダッシュボード", "🏠"),
            ("pages/8_Short_Cover_Hunter.py", "Entry Hunter", "🎯"),
            ("pages/9_Pre_Trade_Check.py", "購入前チェック", "🛡️"),
            ("pages/10_Trade_Journal.py", "売買日誌", "📓"),
        ],
    },
    {
        "label": "スクリーニング",
        "tone": "purple",
        "pages": [
            ("pages/14_Multiple_Expansion_Hunter.py", "ME Hunter", "🚀"),
            ("pages/15_ME_Daily_Screener.py", "MEデイリースクリーナー", "🔎"),
        ],
    },
    {
        "label": "検証・較正",
        "tone": "orange",
        "pages": [
            ("pages/11_Score_Calibration.py", "スコア較正", "⚙️"),
            ("pages/12_Calibration_Backtest.py", "較正バックテスト", "📊"),
            ("pages/13_Walk_Forward_Calibration.py", "ウォークフォワード較正", "📈"),
        ],
    },
    {
        "label": "レポート",
        "tone": "teal",
        "pages": [
            ("pages/📖_レポートビューア.py", "レポートビューア", "📖"),
        ],
    },
]


_SIDEBAR_CSS = """
<style>
/* Hide Streamlit's raw filename navigation and replace it with our app menu. */
[data-testid="stSidebarNav"] {
    display: none;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(59, 130, 246, 0.10), transparent 28rem),
        #f8fafc;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.6rem;
}

.takasan-brand {
    margin: 0.15rem 0 1rem 0;
    padding: 1rem 1rem 0.9rem 1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f3d91 0%, #2563eb 62%, #4f46e5 100%);
    color: white;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.20);
}

.takasan-brand-title {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    line-height: 1.35;
}

.takasan-brand-subtitle {
    margin-top: 0.2rem;
    color: rgba(255, 255, 255, 0.76);
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
}

.takasan-flow-badge {
    margin-top: 0.65rem;
    padding: 0.38rem 0.55rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.takasan-section {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.05rem 0 0.35rem 0.12rem;
    color: #475569;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}

.takasan-section::before {
    content: "";
    width: 5px;
    height: 16px;
    border-radius: 999px;
    background: var(--section-color);
}

.takasan-section.blue { --section-color: #2563eb; }
.takasan-section.purple { --section-color: #7c3aed; }
.takasan-section.orange { --section-color: #f59e0b; }
.takasan-section.teal { --section-color: #0d9488; }

[data-testid="stSidebar"] [data-testid="stPageLink"] {
    margin: 0.22rem 0;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    min-height: 2.8rem;
    padding: 0.58rem 0.72rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 13px;
    background: rgba(255, 255, 255, 0.80);
    color: #172554;
    font-weight: 650;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
    transition:
        background 120ms ease,
        border-color 120ms ease,
        box-shadow 120ms ease,
        transform 120ms ease;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: #ffffff;
    border-color: rgba(59, 130, 246, 0.30);
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.09);
    transform: translateX(2px);
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    position: relative;
    background: linear-gradient(90deg, #eaf2ff 0%, #f5f8ff 100%);
    border-color: rgba(37, 99, 235, 0.28);
    color: #1553c7;
    font-weight: 800;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.12);
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"]::before {
    content: "";
    position: absolute;
    left: -1px;
    top: 8px;
    bottom: 8px;
    width: 4px;
    border-radius: 999px;
    background: #2563eb;
}

.takasan-sidebar-footer {
    margin: 1rem 0 0.35rem 0;
    padding-top: 0.7rem;
    border-top: 1px solid rgba(148, 163, 184, 0.25);
    color: #94a3b8;
    font-size: 0.66rem;
    line-height: 1.5;
}
</style>
"""


def render_sidebar_navigation() -> None:
    """Render the common Japanese sidebar used by every Streamlit page."""
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <div class="takasan-brand">
              <div class="takasan-brand-title">📈 たかさん株式分析</div>
              <div class="takasan-brand-subtitle">TAKASAN STOCK ANALYZER</div>
              <div class="takasan-flow-badge">ME → ENTRY → PRE-TRADE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for section in NAV_SECTIONS:
            st.markdown(
                (
                    f'<div class="takasan-section {section["tone"]}">'
                    f'{section["label"]}</div>'
                ),
                unsafe_allow_html=True,
            )
            for path, label, icon in section["pages"]:
                st.page_link(path, label=label, icon=icon)

        st.markdown(
            """
            <div class="takasan-sidebar-footer">
              DISCOVER → VALIDATE → EXECUTE → REVIEW
            </div>
            """,
            unsafe_allow_html=True,
        )
