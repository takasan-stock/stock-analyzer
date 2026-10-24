from __future__ import annotations

import streamlit as st

from site_context import (
    get_active_stock,
    load_portfolio_stock_options,
    set_active_stock,
    set_active_stock_from_widget,
)


NAV_SECTIONS = [
    {
        "label": "ホーム",
        "tone": "blue",
        "pages": [
            ("dashboard_app.py", "統合ダッシュボード", "🏠"),
            ("pages/16_Stock_Detail.py", "銘柄詳細", "🔬"),
        ],
    },
    {
        "label": "発見・スクリーニング",
        "tone": "purple",
        "pages": [
            ("pages/14_Multiple_Expansion_Hunter.py", "ME Hunter", "🚀"),
            ("pages/15_ME_Daily_Screener.py", "MEデイリースクリーナー", "🔎"),
        ],
    },
    {
        "label": "監視・エントリー",
        "tone": "green",
        "pages": [
            ("pages/8_Short_Cover_Hunter.py", "Entry Hunter", "🎯"),
        ],
    },
    {
        "label": "実行・記録",
        "tone": "red",
        "pages": [
            ("pages/9_Pre_Trade_Check.py", "購入前チェック", "🛡️"),
            ("pages/10_Trade_Journal.py", "売買日誌", "📓"),
        ],
    },
    {
        "label": "分析・レポート",
        "tone": "teal",
        "pages": [
            ("pages/📖_レポートビューア.py", "レポートビューア", "📖"),
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
    color: rgba(255, 255, 255, 0.78);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.13em;
}

.takasan-flow-badge {
    margin-top: 0.65rem;
    padding: 0.38rem 0.55rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.01em;
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
.takasan-section.green { --section-color: #059669; }
.takasan-section.red { --section-color: #e11d48; }
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

/* Main-page site portal */
.takasan-site-hero {
    margin: 0 0 1rem 0;
    padding: 1.1rem 1.25rem;
    border: 1px solid rgba(37, 99, 235, 0.16);
    border-radius: 18px;
    background:
        radial-gradient(circle at 90% 0%, rgba(99, 102, 241, 0.16), transparent 20rem),
        linear-gradient(135deg, #f8fbff 0%, #ffffff 52%, #f7f8ff 100%);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
}

.takasan-site-kicker {
    color: #2563eb;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
}

.takasan-site-title {
    margin-top: 0.15rem;
    color: #0f172a;
    font-size: 1.65rem;
    font-weight: 850;
    letter-spacing: -0.02em;
}

.takasan-site-copy {
    margin-top: 0.3rem;
    color: #64748b;
    font-size: 0.88rem;
    line-height: 1.7;
}

.takasan-site-flow {
    display: inline-block;
    margin-top: 0.65rem;
    padding: 0.38rem 0.7rem;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 0.72rem;
    font-weight: 800;
}

main [data-testid="stPageLink"] a {
    min-height: 2.7rem;
    border-radius: 10px;
    font-weight: 700;
}
</style>
"""


def render_sidebar_navigation() -> None:
    """Render the common site navigation used by every Streamlit page."""
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <div class="takasan-brand">
              <div class="takasan-brand-title">📈 たかさん日本株分析 V3</div>
              <div class="takasan-brand-subtitle">TAKASAN STOCK ANALYZER</div>
              <div class="takasan-flow-badge">DISCOVER → MONITOR → EXECUTE → REVIEW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _stock_options = load_portfolio_stock_options()
        if _stock_options:
            _codes = [code for code, _ in _stock_options]
            _names = {code: name for code, name in _stock_options}
            _active_ticker, _, _ = get_active_stock()
            _active_index = _codes.index(_active_ticker) if _active_ticker in _codes else 0
            if _active_ticker in _codes:
                st.session_state["site_sidebar_stock_selector"] = _active_ticker

            st.markdown(
                '<div class="takasan-section green">現在の銘柄</div>',
                unsafe_allow_html=True,
            )
            _selected_code = st.selectbox(
                "サイト共通の銘柄",
                options=_codes,
                index=_active_index,
                format_func=lambda code: (
                    f"{code}｜{_names.get(code, '')}"
                    if _names.get(code, "")
                    else code
                ),
                key="site_sidebar_stock_selector",
                label_visibility="collapsed",
                on_change=set_active_stock_from_widget,
                kwargs={
                    "widget_key": "site_sidebar_stock_selector",
                    "source": "sidebar",
                    "name_map": _names,
                },
            )
            set_active_stock(
                _selected_code,
                _names.get(_selected_code, ""),
                source="sidebar",
            )
            st.caption("各分析ページへこの銘柄を引き継ぎます。")

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
              DISCOVER → MONITOR → EXECUTE → REVIEW → IMPROVE
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_site_portal() -> None:
    """Render a compact V3 home shell; the dashboard below is the daily command center."""
    st.markdown(
        """
        <div class="takasan-site-hero">
          <div class="takasan-site-kicker">TAKASAN STOCK ANALYZER V3</div>
          <div class="takasan-site-title">📈 たかさん日本株分析</div>
          <div class="takasan-site-copy">
            発見 → 監視 → 実行 → 記録 → 改善を、ひとつのサイトでつなぐ投資ワークスペース。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🧭 分析ツール一覧", expanded=False):
        st.caption("必要な専門画面へ直接移動できます。現在の銘柄はページ間で引き継がれます。")
        c0, c1, c2, c3, c4 = st.columns(5)

        with c0:
            st.markdown("**🔬 銘柄詳細**")
            st.page_link(
                "pages/16_Stock_Detail.py",
                label="詳細を開く",
                icon="🔬",
                use_container_width=True,
            )

        with c1:
            st.markdown("**🔎 発見**")
            st.page_link(
                "pages/15_ME_Daily_Screener.py",
                label="ME Daily",
                icon="🔎",
                use_container_width=True,
            )
            st.page_link(
                "pages/14_Multiple_Expansion_Hunter.py",
                label="ME Hunter",
                icon="🚀",
                use_container_width=True,
            )

        with c2:
            st.markdown("**🎯 監視**")
            st.page_link(
                "pages/8_Short_Cover_Hunter.py",
                label="Entry Hunter",
                icon="🎯",
                use_container_width=True,
            )

        with c3:
            st.markdown("**🛡️ 実行**")
            st.page_link(
                "pages/9_Pre_Trade_Check.py",
                label="購入前チェック",
                icon="🛡️",
                use_container_width=True,
            )
            st.page_link(
                "pages/10_Trade_Journal.py",
                label="売買日誌",
                icon="📓",
                use_container_width=True,
            )

        with c4:
            st.markdown("**📚 記録・検証**")
            st.page_link(
                "pages/📖_レポートビューア.py",
                label="レポート",
                icon="📖",
                use_container_width=True,
            )
            st.page_link(
                "pages/11_Score_Calibration.py",
                label="スコア較正",
                icon="⚙️",
                use_container_width=True,
            )

        with st.expander("🧪 詳細な検証ツール", expanded=False):
            v1, v2 = st.columns(2)
            with v1:
                st.page_link(
                    "pages/12_Calibration_Backtest.py",
                    label="較正バックテスト",
                    icon="📊",
                    use_container_width=True,
                )
            with v2:
                st.page_link(
                    "pages/13_Walk_Forward_Calibration.py",
                    label="ウォークフォワード較正",
                    icon="📈",
                    use_container_width=True,
                )
