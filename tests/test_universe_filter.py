import pandas as pd

from data.universe import _is_common_stock


def test_common_stock_filter_basic():
    row = pd.Series({"Symbol": "AAPL", "ETF": "N", "Test Issue": "N"})
    assert _is_common_stock(row, include_etf=False) is True


def test_common_stock_filter_excludes_etf_and_test_issue():
    etf_row = pd.Series({"Symbol": "QQQ", "ETF": "Y", "Test Issue": "N"})
    test_row = pd.Series({"Symbol": "ZZZZ", "ETF": "N", "Test Issue": "Y"})
    assert _is_common_stock(etf_row, include_etf=False) is False
    assert _is_common_stock(test_row, include_etf=False) is False
