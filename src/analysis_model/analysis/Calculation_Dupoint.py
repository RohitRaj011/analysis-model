"""DuPont 5-Step Formula Calculation Module.

This module processes cleaned financial statements (P&L, Balance Sheet, Cash Flow)
and computes the core 5 ratios required for DuPont Return on Equity (ROE) decomposition
across three consecutive fiscal years.

Financial Math Reference:
    1. Operating Margin = EBIT / Revenue
    2. Asset Turnover   = Revenue / Average Total Assets
    3. Equity Multiplier = Average Total Assets / Average Total Equity
    4. Tax Burden       = Net Income / EBT
    5. Interest Burden   = EBT / EBIT
"""

from typing import Optional, Any
import pandas as pd

# Forward reference / type hinting import for StoringAndCleaning
from analysis_model.data.StoringandCleaning import StoringAndCleaning


class Calculation:
    """Computes historical 5-step DuPont analysis ratios from cleaned financial DataFrames.

    Attributes:
        pnl (pd.DataFrame): Sorted Income Statement DataFrame indexed by fiscal year.
        bs (pd.DataFrame): Sorted Balance Sheet DataFrame indexed by fiscal year.
        cf (pd.DataFrame): Sorted Cash Flow DataFrame indexed by fiscal year.
        extra (pd.DataFrame): Additional auxiliary market or macro metrics DataFrame.
        years (pd.Index): Index container holding available fiscal years.
        latest_year (int): Target recent fiscal year (T).
        previous_year (int): Prior fiscal year (T-1).
        third_year (int): Three fiscal years ago (T-2).
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Extracts financial DataFrames from the storage layer and prepares rolling metrics.

        Design Rationale:
            Balance sheet metrics (Total Assets, Equity) are point-in-time snapshots.
            To accurately align them with annual flow metrics (Revenue, Net Income),
            we compute 2-period rolling averages (`rolling(window=2).mean()`).

        Args:
            fetcher (StoringAndCleaning): Provider holding parsed Pandas DataFrames.
        """
        self.pnl: pd.DataFrame = fetcher.df_pnl.sort_index(ascending=True)
        self.bs: pd.DataFrame = fetcher.df_bs.sort_index(ascending=True)
        self.cf: pd.DataFrame = fetcher.df_cf.sort_index(ascending=True)
        self.extra: pd.DataFrame = fetcher.df_extra

        self.years: pd.Index = self.bs.index
        self.latest_year: int = int(max(self.years))
        self.previous_year: int = self.latest_year - 1
        self.third_year: int = self.latest_year - 2

        # Transform point-in-time Balance Sheet figures into operating-year averages
        self.bs["Average asset"] = self.bs["Total Assets"].rolling(window=2).mean()
        self.bs["Average Equity"] = self.bs["Total Equity"].rolling(window=2).mean()

        # --- Year T (Latest Year) Ratio Placeholders ---
        self.net_profit_margin_latest_year: Optional[float] = None
        self.asset_turnover_latest_year: Optional[float] = None
        self.equity_multiplier_latest_year: Optional[float] = None
        self.tax_burden_latest_year: Optional[float] = None
        self.interest_burden_latest_year: Optional[float] = None

        # --- Year T-1 (Previous Year) Ratio Placeholders ---
        self.net_profit_margin_previous_year: Optional[float] = None
        self.asset_turnover_previous_year: Optional[float] = None
        self.equity_multiplier_previous_year: Optional[float] = None
        self.tax_burden_previous_year: Optional[float] = None
        self.interest_burden_previous_year: Optional[float] = None

        # --- Year T-2 (Third Year) Ratio Placeholders ---
        self.net_profit_margin_third_year: Optional[float] = None
        self.asset_turnover_third_year: Optional[float] = None
        self.equity_multiplier_third_year: Optional[float] = None
        self.tax_burden_third_year: Optional[float] = None
        self.interest_burden_third_year: Optional[float] = None

    def master_initializer(self) -> None:
        """Triggers the calculation of all 5 DuPont sub-ratios for the 3 target years."""
        # 1. Latest Fiscal Year Calculations (T)
        self.net_profit_margin_latest_year = self.operating_margin_calc(self.latest_year)
        self.asset_turnover_latest_year = self.asset_turnover_calc(self.latest_year)
        self.equity_multiplier_latest_year = self.equity_multiplier_calc(self.latest_year)
        self.tax_burden_latest_year = self.tax_burden_calc(self.latest_year)
        self.interest_burden_latest_year = self.interest_burden_calc(self.latest_year)

        # 2. Previous Fiscal Year Calculations (T-1)
        prev_year: int = int(self.latest_year - 1)
        self.net_profit_margin_previous_year = self.operating_margin_calc(prev_year)
        self.asset_turnover_previous_year = self.asset_turnover_calc(prev_year)
        self.equity_multiplier_previous_year = self.equity_multiplier_calc(prev_year)
        self.tax_burden_previous_year = self.tax_burden_calc(prev_year)
        self.interest_burden_previous_year = self.interest_burden_calc(prev_year)

        # 3. Third Fiscal Year Calculations (T-2)
        third_year: int = int(self.latest_year - 2)
        self.net_profit_margin_third_year = self.operating_margin_calc(third_year)
        self.asset_turnover_third_year = self.asset_turnover_calc(third_year)
        self.equity_multiplier_third_year = self.equity_multiplier_calc(third_year)
        self.tax_burden_third_year = self.tax_burden_calc(third_year)
        self.interest_burden_third_year = self.interest_burden_calc(third_year)

    def operating_margin_calc(self, year: int) -> float:
        """Calculates Operating Margin (EBIT / Revenue) for a target year.

        Args:
            year (int): Fiscal year to evaluate.

        Returns:
            float: Computed operating margin, or 0.0 if revenue is zero.
        """
        revenue: float = float(self.pnl.loc[year, "Revenue"])
        ebit: float = float(self.pnl.loc[year, "Ebit"])
        return ebit / revenue if revenue else 0.0

    def asset_turnover_calc(self, year: int) -> float:
        """Calculates Asset Turnover (Revenue / Average Assets) for a target year.

        Args:
            year (int): Fiscal year to evaluate.

        Returns:
            float: Computed asset turnover ratio, or 0.0 if average asset is zero.
        """
        revenue: float = float(self.pnl.loc[year, "Revenue"])
        avg_asset: float = float(self.bs.loc[year, "Average asset"])
        return revenue / avg_asset if avg_asset else 0.0

    def equity_multiplier_calc(self, year: int) -> float:
        """Calculates Equity Multiplier (Average Assets / Average Equity) for a target year.

        Args:
            year (int): Fiscal year to evaluate.

        Returns:
            float: Leverage ratio, or 0.0 if average equity is zero.
        """
        avg_asset: float = float(self.bs.loc[year, "Average asset"])
        avg_equity: float = float(self.bs.loc[year, "Average Equity"])
        return avg_asset / avg_equity if avg_equity else 0.0

    def tax_burden_calc(self, year: int) -> float:
        """Calculates Tax Burden Ratio (Net Income / EBT) for a target year.

        Args:
            year (int): Fiscal year to evaluate.

        Returns:
            float: Tax retention ratio, or 0.0 if EBT (Income Before Tax) is zero.
        """
        net_income: float = float(self.pnl.loc[year, "Net Income"])
        ebt: float = float(self.pnl.loc[year, "Income Before Tax"])
        return net_income / ebt if ebt else 0.0

    def interest_burden_calc(self, year: int) -> float:
        """Calculates Interest Burden Ratio (EBT / EBIT) for a target year.

        Args:
            year (int): Fiscal year to evaluate.

        Returns:
            float: Interest retention ratio, or 0.0 if EBIT is zero.
        """
        ebt: float = float(self.pnl.loc[year, "Income Before Tax"])
        ebit: float = float(self.pnl.loc[year, "Ebit"])
        return ebt / ebit if ebit else 0.0
