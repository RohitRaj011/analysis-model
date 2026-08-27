from types import SimpleNamespace

import pandas as pd
import pytest

from analysis_model.analysis.Calculation_Altman_zscore import Calculation as ZCalc
from analysis_model.analysis.Calculation_Dupoint import Calculation as DupontCalc
from analysis_model.analysis.Calculations_CCC import Calculation as CCCCalc
from analysis_model.analysis.Financial_health import FinancialHealth
from analysis_model.analysis.Return_on_capital import ReturnOnCapital
from analysis_model.analysis.operational_efficency import OperationEFFICIENCY
from analysis_model.analysis.valuation import Valuation
from analysis_model.data.StoringandCleaning import match_equity_risk_premium
from analysis_model.pipeline import short_dcf_verdict


def _storing(
    years,
    pnl_rows,
    bs_rows,
    extra_row,
    country="United States",
    sector="industrials",
):
    fetcher = SimpleNamespace(
        df_pnl=pd.DataFrame(pnl_rows, index=years),
        df_bs=pd.DataFrame(bs_rows, index=years),
        df_cf=pd.DataFrame(
            {
                "Cashflow from operation": [80, 90, 100],
                "Capex": [-20, -20, -20],
                "Accounts Receivables": [1, 1, 1],
            },
            index=years,
        ),
        df_extra=pd.DataFrame(extra_row),
        country=country,
        sector=sector,
    )
    return fetcher


def test_altman_zone_thresholds_manufacturing():
    ratios = SimpleNamespace(
        x1=0.2, x2=0.2, x3=0.2, x4=1.0, x5=0.5,
        sector="industrials", country="germany",
    )
    health = FinancialHealth(fetcher=ratios)
    health.fetch_all_data()
    assert health.lst_zone == [1.23, 2.9]
    assert FinancialHealth.uses_x5_ratio("industrials", "germany") is True


def test_altman_us_does_not_use_x5():
    assert FinancialHealth.uses_x5_ratio("industrials", "United States") is False
    ratios = SimpleNamespace(
        x1=0.1, x2=0.1, x3=0.1, x4=0.5, x5=9.0,
        sector="industrials", country="United States",
    )
    health = FinancialHealth(fetcher=ratios)
    health.score_calc()
    without_x5 = 3.25 + 6.56 * 0.1 + 3.26 * 0.1 + 6.72 * 0.1 + 1.05 * 0.5
    assert health.score == pytest.approx(without_x5)


def test_altman_ratios_from_statements():
    years = [2021, 2022, 2023]
    pnl = {
        "Revenue": [100, 110, 200],
        "Ebit": [10, 11, 40],
        "Interest Expenses": [1, 1, 1],
        "Tax": [2, 2, 4],
        "Operating Income": [10, 11, 40],
        "Income Before Tax": [9, 10, 36],
        "Depreciation And Amortization": [3, 3, 3],
        "Net Income": [7, 8, 30],
        "Cost of Goods Sold": [50, 55, 80],
    }
    bs = {
        "Total Debt": [20, 20, 20],
        "Cash": [5, 5, 5],
        "Accounts Receivable": [10, 10, 10],
        "Inventory": [8, 8, 8],
        "Accounts Payable": [4, 4, 4],
        "Total Equity": [50, 55, 60],
        "Total Assets": [100, 100, 100],
        "Retained Earnings": [20, 25, 30],
        "Working Capital": [10, 12, 20],
        "Total Liabilities": [50, 45, 40],
    }
    extra = {
        "Current Price": [10],
        "Beta": [1.1],
        "Market Captilization": [80],
        "Bond Interest Rate": [4],
        "Dilluted Shares": [8],
        "Equity Premium": [5],
    }
    calc = ZCalc(fetcher=_storing(years, pnl, bs, extra))
    calc.initializer()
    assert calc.x1 == pytest.approx(20 / 100)
    assert calc.x2 == pytest.approx(30 / 100)
    assert calc.x3 == pytest.approx(40 / 100)
    assert calc.x4 == pytest.approx(80 / 40)
    assert calc.x5 == pytest.approx(200 / 100)


def test_dupont_product_identity():
    assert ReturnOnCapital.dupoint_calc(0.2, 1.5, 2.0, 0.8, 0.9) == pytest.approx(
        0.2 * 1.5 * 2.0 * 0.8 * 0.9
    )


def test_dupont_from_statements():
    years = [2021, 2022, 2023]
    pnl = {
        "Revenue": [100, 120, 150],
        "Ebit": [20, 24, 30],
        "Interest Expenses": [2, 2, 2],
        "Tax": [4, 5, 6],
        "Operating Income": [20, 24, 30],
        "Income Before Tax": [18, 22, 28],
        "Depreciation And Amortization": [3, 3, 3],
        "Net Income": [14, 17, 22],
        "Cost of Goods Sold": [50, 60, 70],
    }
    bs = {
        "Total Debt": [20, 20, 20],
        "Cash": [5, 5, 5],
        "Accounts Receivable": [10, 12, 14],
        "Inventory": [8, 9, 10],
        "Accounts Payable": [4, 5, 6],
        "Total Equity": [40, 50, 60],
        "Total Assets": [80, 100, 120],
        "Retained Earnings": [10, 15, 20],
        "Working Capital": [10, 12, 14],
        "Total Liabilities": [40, 50, 60],
    }
    extra = {"Current Price": [1], "Beta": [1], "Market Captilization": [1],
             "Bond Interest Rate": [1], "Dilluted Shares": [1], "Equity Premium": [1]}
    calc = DupontCalc(fetcher=_storing(years, pnl, bs, extra))
    calc.master_initializer()
    roc = ReturnOnCapital(fetcher=calc)
    roc.calc()
    expected = (
        calc.net_profit_margin_latest_year
        * calc.asset_turnover_latest_year
        * calc.equity_multiplier_latest_year
        * calc.tax_burden_latest_year
        * calc.interest_burden_latest_year
    )
    assert roc.dupoint_analysis_latest_year == pytest.approx(expected)


def test_ccc_identity_and_statements():
    assert OperationEFFICIENCY.cash_conversion_cycle(40, 30, 25) == pytest.approx(45)
    years = [2021, 2022, 2023]
    pnl = {
        "Revenue": [365, 365, 365],
        "Ebit": [10, 10, 10],
        "Interest Expenses": [1, 1, 1],
        "Tax": [1, 1, 1],
        "Operating Income": [10, 10, 10],
        "Income Before Tax": [9, 9, 9],
        "Depreciation And Amortization": [1, 1, 1],
        "Net Income": [8, 8, 8],
        "Cost of Goods Sold": [365, 365, 365],
    }
    bs = {
        "Total Debt": [1, 1, 1],
        "Cash": [1, 1, 1],
        "Accounts Receivable": [10, 10, 10],
        "Inventory": [20, 20, 20],
        "Accounts Payable": [5, 5, 5],
        "Total Equity": [1, 1, 1],
        "Total Assets": [50, 50, 50],
        "Retained Earnings": [1, 1, 1],
        "Working Capital": [1, 1, 1],
        "Total Liabilities": [1, 1, 1],
    }
    extra = {"Current Price": [1], "Beta": [1], "Market Captilization": [1],
             "Bond Interest Rate": [1], "Dilluted Shares": [1], "Equity Premium": [1]}
    calc = CCCCalc(fetcher=_storing(years, pnl, bs, extra))
    calc.master_initializer()
    op = OperationEFFICIENCY(fetcher=calc)
    op.calc()
    assert op.ccc_latest_year == pytest.approx(
        op.daily_inventory_outstanding_latest_year
        + op.daily_sales_outstanding_latest_year
        - op.daily_payable_outstanding_latest_year
    )


def test_dcf_equity_bridge():
    calc = SimpleNamespace(
        enterprise_value=100.0,
        latest_year=2023,
        bs=pd.DataFrame({"Cash": [15.0], "Total Debt": [25.0]}, index=[2023]),
        extra=pd.DataFrame({"Dilluted Shares": [10.0], "Current Price": [8.0]}),
    )
    val = Valuation(fetcher=calc)
    val.calc()
    assert val.equity_value == pytest.approx(90.0)
    assert val.intrisnic_value == pytest.approx(9.0)
    assert "undervalued" in val.value.lower()
    assert short_dcf_verdict(val.value) == "Undervalued"


def test_erp_matches_country_not_fixed_index():
    rows = [
        {"country": "Argentina", "totalEquityRiskPremium": 15.0},
        {"country": "Germany", "totalEquityRiskPremium": 5.5},
        {"country": "United States", "totalEquityRiskPremium": 4.6},
    ]
    rate, warning = match_equity_risk_premium(rows, "Germany")
    assert rate == pytest.approx(5.5)
    assert warning is None
    rate_us, warning_us = match_equity_risk_premium(rows, "Narnia")
    assert rate_us == pytest.approx(4.6)
    assert warning_us is not None
