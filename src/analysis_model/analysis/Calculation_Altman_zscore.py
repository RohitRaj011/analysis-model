"""Altman Z-Score Metric Calculation Module.

This module processes financial balance sheet, income statement, and market data DataFrames 
to derive the five core financial ratios ($X_1$ through $X_5$) required for 
Edward Altman's Z-Score bankruptcy prediction model.

Financial Ratio Reference:
    - X1 = Working Capital / Total Assets       (Measures short-term liquidity)
    - X2 = Retained Earnings / Total Assets    (Measures cumulative profitability/reinvested leverage)
    - X3 = EBIT / Total Assets                 (Measures true asset productivity/operating efficiency)
    - X4 = Market Cap / Total Liabilities       (Measures solvency and market equity buffer)
    - X5 = Revenue / Total Assets              (Measures capital turnover/asset utilization)
"""

from typing import Optional
import pandas as pd

# Type hinting reference for StoringAndCleaning
from analysis_model.data.StoringandCleaning import StoringAndCleaning


class Calculation:
    """Computes the five fundamental financial ratios ($X_1 - X_5$) for Altman Z-Score analysis.

    Attributes:
        bs_df (pd.DataFrame): Sorted Balance Sheet DataFrame indexed by fiscal year.
        pnl_df (pd.DataFrame): Sorted Income Statement DataFrame indexed by fiscal year.
        cf_df (pd.DataFrame): Sorted Cash Flow Statement DataFrame indexed by fiscal year.
        extra_df (pd.DataFrame): Auxiliary DataFrame containing real-time market data (e.g., Market Cap).
        years (pd.Index): Index container of available fiscal years.
        latest_year (int): Most recent fiscal year selected for calculation.
        country (str): Target stock's country of incorporation.
        sector (str): Target stock's industry sector.
        total_assets (Optional[float]): Cached total assets value used across multiple ratio denominators.
        x1 (Optional[float]): Working Capital / Total Assets.
        x2 (Optional[float]): Retained Earnings / Total Assets.
        x3 (Optional[float]): EBIT / Total Assets.
        x4 (Optional[float]): Market Capitalization / Total Liabilities.
        x5 (Optional[float]): Revenue / Total Assets.
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Extracts financial DataFrames and sets metadata context for ratio evaluation.

        Args:
            fetcher (StoringAndCleaning): Cleaned financial data provider containing parsed DataFrames.
        """
        self.bs_df: pd.DataFrame = fetcher.df_bs.sort_index(ascending=True)
        self.pnl_df: pd.DataFrame = fetcher.df_pnl.sort_index(ascending=True)
        self.cf_df: pd.DataFrame = fetcher.df_cf.sort_index(ascending=True)
        self.extra_df: pd.DataFrame = fetcher.df_extra

        self.years: pd.Index = fetcher.df_bs.index
        self.latest_year: int = int(max(self.years))

        # Metadata used downstream to adjust coefficient weights (e.g., manufacturing vs non-manufacturing)
        self.country: str = str(fetcher.country)
        self.sector: str = str(fetcher.sector)

        # Common denominator state
        self.total_assets: Optional[float] = None

        # Z-Score variable placeholders
        self.x1: Optional[float] = None
        self.x2: Optional[float] = None
        self.x3: Optional[float] = None
        self.x4: Optional[float] = None
        self.x5: Optional[float] = None

    def initializer(self) -> None:
        """Triggers execution of all five Z-score variable calculation routines for the latest fiscal year."""
        self.x1_calc()
        self.x2_calc()
        self.x3_calc()
        self.x4_calc()
        self.x5_calc()

    def x1_calc(self) -> None:
        """Calculates $X_1$: Working Capital / Total Assets.

        Design Rationale:
            Caches `self.total_assets` during this step so that subsequent ratio methods 
            ($X_2, X_3, X_5$) can reuse the denominator without repeated DataFrame lookups.
        """
        working_capital: float = float(self.bs_df.loc[self.latest_year, "Working Capital"])
        self.total_assets = float(self.bs_df.loc[self.latest_year, "Total Assets"])
        
        self.x1 = working_capital / self.total_assets if self.total_assets else 0.0

    def x2_calc(self) -> None:
        """Calculates $X_2$: Retained Earnings / Total Assets.

        Measures cumulative profitability over time; younger firms naturally have lower $X_2$ values.
        """
        retained_earnings: float = float(self.bs_df.loc[self.latest_year, "Retained Earnings"])
        
        if self.total_assets:
            self.x2 = retained_earnings / self.total_assets
        else:
            self.x2 = 0.0

    def x3_calc(self) -> None:
        """Calculates $X_3$: EBIT / Total Assets.

        Measures operating productivity of assets unburdened by tax or leverage factors.
        """
        ebit: float = float(self.pnl_df.loc[self.latest_year, "Ebit"])
        
        if self.total_assets:
            self.x3 = ebit / self.total_assets
        else:
            self.x3 = 0.0

    def x4_calc(self) -> None:
        """Calculates $X_4$: Market Capitalization / Total Liabilities.

        Measures how much the firm's equity market value can decline before liabilities exceed assets.
        Note: Reads `Market Capitalization` from `extra_df` (sourced via market ticker API).
        """
        market_cap: float = float(self.extra_df.loc[0, "Market Captilization"])
        total_liabilities: float = float(self.bs_df.loc[self.latest_year, "Total Liabilities"])
        
        self.x4 = market_cap / total_liabilities if total_liabilities else 0.0

    def x5_calc(self) -> None:
        """Calculates $X_5$: Sales (Revenue) / Total Assets.

        Measures total sales-generating capacity of company assets (asset turnover).
        """
        sales: float = float(self.pnl_df.loc[self.latest_year, "Revenue"])
        
        if self.total_assets:
            self.x5 = sales / self.total_assets
        else:
            self.x5 = 0.0
