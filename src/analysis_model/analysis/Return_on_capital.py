import pandas as pd
from typing import Dict, Optional, Protocol, Any, Tuple


class DuPontFetcherProtocol(Protocol):
    """
    Protocol defining the expected interface for financial data inputs
    required to compute multi-year DuPont decomposition ratios.
    """
    net_profit_margin_latest_year: float
    asset_turnover_latest_year: float
    equity_multiplier_latest_year: float
    tax_burden_latest_year: float
    interest_burden_latest_year: float

    net_profit_margin_previous_year: float
    asset_turnover_previous_year: float
    equity_multiplier_previous_year: float
    tax_burden_previous_year: float
    interest_burden_previous_year: float

    net_profit_margin_third_year: float
    asset_turnover_third_year: float
    equity_multiplier_third_year: float
    tax_burden_third_year: float
    interest_burden_third_year: float

    latest_year: int
    previous_year: int
    third_year: int


class ReturnOnCapital:
    """
    Computes extended 5-way DuPont Analysis across a 3-year historical window.

    The 5-Step DuPont Analysis breaks down Return on Equity (ROE) into operational,
    financial, tax, and interest efficiency components to isolate driver trends over time.

    Formula
    -------
    $$\\text{ROE} = \\text{Tax Burden} \\times \\text{Interest Burden} \\times 
    \\text{EBIT Margin} \\times \\text{Asset Turnover} \\times \\text{Equity Multiplier}$$
    """

    def __init__(self, fetcher: Any) -> None:
        """
        Initializes multi-year financial ratio components from the data fetcher.

        Parameters
        ----------
        fetcher : DuPontFetcherProtocol
            Data container providing raw line items and ratios across three consecutive years.
        """
        # Metadata / Year Identifiers
        self.latest_year: int = int(fetcher.latest_year)
        self.previous_year: int = int(fetcher.previous_year)
        self.third_year: int = int(fetcher.third_year)

        # Latest Year Ratio Components
        self.net_profit_margin_latest_year: float = float(fetcher.net_profit_margin_latest_year)
        self.asset_turnover_latest_year: float = float(fetcher.asset_turnover_latest_year)
        self.equity_multiplier_latest_year: float = float(fetcher.equity_multiplier_latest_year)
        self.tax_burden_latest_year: float = float(fetcher.tax_burden_latest_year)
        self.interest_burden_latest_year: float = float(fetcher.interest_burden_latest_year)

        # Previous Year Ratio Components
        self.net_profit_margin_previous_year: float = float(fetcher.net_profit_margin_previous_year)
        self.asset_turnover_previous_year: float = float(fetcher.asset_turnover_previous_year)
        self.equity_multiplier_previous_year: float = float(fetcher.equity_multiplier_previous_year)
        self.tax_burden_previous_year: float = float(fetcher.tax_burden_previous_year)
        self.interest_burden_previous_year: float = float(fetcher.interest_burden_previous_year)

        # Third Year Ratio Components
        self.net_profit_margin_third_year: float = float(fetcher.net_profit_margin_third_year)
        self.asset_turnover_third_year: float = float(fetcher.asset_turnover_third_year)
        self.equity_multiplier_third_year: float = float(fetcher.equity_multiplier_third_year)
        self.tax_burden_third_year: float = float(fetcher.tax_burden_third_year)
        self.interest_burden_third_year: float = float(fetcher.interest_burden_third_year)

        # Computed ROE Outputs
        self.dupoint_analysis_latest_year: Optional[float] = None
        self.dupoint_analysis_previous_year: Optional[float] = None
        self.dupoint_analysis_third_year: Optional[float] = None

    def calc(self) -> Tuple[float, float, float]:
        """
        Executes the DuPont calculation across all three distinct historical years.

        Returns
        -------
        Tuple[float, float, float]
            Calculated ROE values for `(latest_year, previous_year, third_year)`.
        """
        self.dupoint_analysis_latest_year = self.dupoint_calc(
            npm=self.net_profit_margin_latest_year,
            at=self.asset_turnover_latest_year,
            em=self.equity_multiplier_latest_year,
            tb=self.tax_burden_latest_year,
            ib=self.interest_burden_latest_year,
        )

        self.dupoint_analysis_previous_year = self.dupoint_calc(
            npm=self.net_profit_margin_previous_year,
            at=self.asset_turnover_previous_year,
            em=self.equity_multiplier_previous_year,
            tb=self.tax_burden_previous_year,
            ib=self.interest_burden_previous_year,
        )

        self.dupoint_analysis_third_year = self.dupoint_calc(
            npm=self.net_profit_margin_third_year,
            at=self.asset_turnover_third_year,
            em=self.equity_multiplier_third_year,
            tb=self.tax_burden_third_year,
            ib=self.interest_burden_third_year,
        )

        return (
            self.dupoint_analysis_latest_year,
            self.dupoint_analysis_previous_year,
            self.dupoint_analysis_third_year,
        )

    @staticmethod
    def dupoint_calc(npm: float, at: float, em: float, tb: float, ib: float) -> float:
        """
        Multiplies the 5 drivers of DuPont decomposition to calculate Return on Equity (ROE).

        Parameters
        ----------
        npm : float
            Net Profit Margin (EBIT / Revenue).
        at : float
            Asset Turnover (Revenue / Total Assets).
        em : float
            Equity Multiplier (Total Assets / Total Equity).
        tb : float
            Tax Burden (Net Income / EBT).
        ib : float
            Interest Burden (EBT / EBIT).

        Returns
        -------
        float
            Decomposed Return on Equity (ROE) metric.
        """
        return npm * at * em * tb * ib
