import pandas as pd

from portfolio_quality import (
    build_company_news_query,
    display_portfolio_dataframe,
    find_duplicate_tickers,
    is_relevant_company_news,
    normalize_metric_text,
    normalize_portfolio_dataframe,
    normalize_ticker_code,
)


def test_metric_normalization_removes_display_units():
    assert normalize_metric_text("38.7%") == "38.7"
    assert normalize_metric_text("7.59％") == "7.59"
    assert normalize_metric_text("15.5倍") == "15.5"
    assert normalize_metric_text("1.33x") == "1.33"
    assert normalize_metric_text(None) == ""


def test_portfolio_dataframe_normalizes_numeric_fields():
    frame = pd.DataFrame(
        [{
            "ティッカー": "472a ",
            "売上5y CAGR": "15.0%",
            "ROIC": "7.20%",
            "PER": "12.0倍",
            "DPUP": "0.79x",
        }]
    )

    out = normalize_portfolio_dataframe(frame)

    assert out.iloc[0]["ティッカー"] == "472A"
    assert out.iloc[0]["売上5y CAGR"] == "15"
    assert out.iloc[0]["ROIC"] == "7.2"
    assert out.iloc[0]["PER"] == "12"
    assert out.iloc[0]["DPUP"] == "0.79"


def test_display_dataframe_formats_percent_and_missing_values():
    frame = pd.DataFrame(
        [{
            "ティッカー": "9560",
            "売上5y CAGR": "36.6",
            "ROIC": "47.80",
            "PER": "13.1",
            "DPUP": "4.24",
            "材料メモ": None,
        }]
    )

    out = display_portfolio_dataframe(frame)

    assert out.iloc[0]["売上5y CAGR"] == "36.6%"
    assert out.iloc[0]["ROIC"] == "47.8%"
    assert out.iloc[0]["PER"] == "13.1"
    assert out.iloc[0]["DPUP"] == "4.24"
    assert out.iloc[0]["材料メモ"] == "—"


def test_duplicate_ticker_detection_normalizes_codes():
    frame = pd.DataFrame(
        {
            "ティッカー": ["9560", "472a", "472A ", "09560", ""],
        }
    )

    assert find_duplicate_tickers(frame) == ["472A"]


def test_company_news_query_contains_name_ticker_and_finance_context():
    query = build_company_news_query("4019", "スタメン")
    assert '"スタメン"' in query
    assert "4019" in query
    assert "決算" in query
    assert "IR" in query


def test_irrelevant_sports_story_is_rejected_for_stamen():
    assert is_relevant_company_news(
        "マンチェスター・ユナイテッド対トッテナム、予想スタメン＆フォーメーション",
        "4019",
        "スタメン",
        "ABEMA",
    ) is False


def test_relevant_company_story_is_kept_for_stamen():
    assert is_relevant_company_news(
        "スタメン、通期業績予想を上方修正　営業利益が拡大",
        "4019",
        "スタメン",
        "日本経済新聞",
    ) is True


def test_ticker_hit_can_disambiguate_generic_company_name():
    assert is_relevant_company_news(
        "5243 NOTE が新サービスを発表",
        "5243",
        "NOTE",
        "Tech News",
    ) is True


def test_ticker_normalization_preserves_alphanumeric_tse_codes():
    assert normalize_ticker_code("472a") == "472A"
    assert normalize_ticker_code("9560.0") == "9560"
