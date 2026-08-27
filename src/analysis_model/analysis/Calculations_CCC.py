"""Cash Conversion Cycle (CCC) Metric Calculation Module.

This module processes cleaned financial statements (P&L and Balance Sheet DataFrames)
to compute working capital efficiency metrics across three consecutive fiscal years:
    - Days Inventory Outstanding (DIO)
    - Days Sales Outstanding (DSO)
    - Days Payable Outstanding (DPO)

Financial Math Reference:
    - DIO = (Average Inventory / Cost of Goods Sold) * 365
    - DSO = (Average Accounts Receivable / Revenue) * 365
    - DPO = (Average Accounts Payable / Cost of Goods Sold) * 365
    - Net CCC = DIO + DSO - DPO (Calculated downstream in OperationEFFICIENCY)
"""

from typing import Optional
import pandas as pd
import numpy as np

# Type hinting reference for StoringAndCleaning
from analysis_model.data.StoringandCleaning import StoringAndCleaning


class Calculation:
    """Computes DIO, DSO, and DPO operational metrics across three historical fiscal years.

    Attributes:
        pnl (pd.DataFrame): Sorted Income Statement DataFrame indexed by fiscal year.
        bs (pd.DataFrame): Sorted Balance Sheet DataFrame indexed by fiscal year.
        cf (pd.DataFrame): Sorted Cash Flow Statement DataFrame indexed by fiscal year.
        extra (pd.DataFrame): Auxiliary market data DataFrame.
        years (pd.Index): Index container holding available fiscal years.
        latest_year (int): Target recent fiscal year (T).
        previous_year (int): Prior fiscal year (T-1).
        third_year (int): Three fiscal years ago (T-2).
        daily_inventory_outstanding_latest_year (Optional[float]): DIO for year T.
        daily_sales_outstanding_latest_year (Optional[float]): DSO for year T.
        daily_payable_outstanding_latest_year (Optional[float]): DPO for year T.
        daily_inventory_outstanding_previous_year (Optional[float]): DIO for year T-1.
        daily_sales_outstanding_previous_year (Optional[float]): DSO for year T-1.
        daily_payable_outstanding_previous_year (Optional[float]): DPO for year T-1.
        daily_inventory_outstanding_third_year (Optional[float]): DIO for year T-2.
        daily_sales_outstanding_third_year (Optional[float]): DSO for year T-2.
        daily_payable_outstanding_third_year (Optional[float]): DPO for year T-2.
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Extracts financial DataFrames from storage and computes 2-period rolling averages.

        Design Rationale:
            Inventory, Accounts Receivable, and Accounts Payable are point-in-time balance sheet figures.
            To accurately align them with annual flow metrics (Revenue, COGS), we compute 2-period 
            rolling averages (`rolling(window=2).mean()`) to avoid single-day seasonal distortion.

        Args:
            fetcher (StoringAndCleaning): Cleaned financial data provider containing parsed DataFrames.
        """
        self.pnl: pd.DataFrame = fetcher.df_pnl.sort_index(ascending=True)
        self.bs: pd.DataFrame = fetcher.df_bs.sort_index(ascending=True)
        self.cf: pd.DataFrame = fetcher.df_cf.sort_index(ascending=True)
        self.extra: pd.DataFrame = fetcher.df_extra

        self.years: pd.Index = self.bs.index
        self.latest_year: int = int(max(self.years))
        self.previous_year: int = self.latest_year - 1
        self.third_year: int = self.latest_year - 2

        # Transform point-in-time working capital items into operating-year averages
        self.bs["Average Inventory"] = self.bs["Inventory"].rolling(window=2).mean()
        self.bs["Average Accounts Receivable"] = (
            self.bs["Accounts Receivable"].rolling(window=2).mean()
        )
        self.bs["Average Accounts Payable"] = (
            self.bs["Accounts Payable"].rolling(window=2).mean()
        )

        # --- Year T (Latest Year) Metric Placeholders ---
        self.daily_inventory_outstanding_latest_year: Optional[float] = None
        self.daily_sales_outstanding_latest_year: Optional[float] = None
        self.daily_payable_outstanding_latest_year: Optional[float] = None

        # --- Year T-1 (Previous Year) Metric Placeholders ---
        self.daily_inventory_outstanding_previous_year: Optional[float] = None
        self.daily_sales_outstanding_previous_year: Optional[float] = None
        self.daily_payable_outstanding_previous_year: Optional[float] = None

        # --- Year T-2 (Third Year) Metric Placeholders ---
        self.daily_inventory_outstanding_third_year: Optional[float] = None
        self.daily_sales_outstanding_third_year: Optional[float] = None
        self.daily_payable_outstanding_third_year: Optional[float] = None

    def master_initializer(self) -> None:
        """Executes DIO, DSO, and DPO calculations across all three target fiscal years."""
        # 1. Latest Fiscal Year Calculations (T)
        self.daily_inventory_outstanding_latest_year = (
            self.daily_inventory_outstanding_calc(self.latest_year)
        )
        self.daily_payable_outstanding_latest_year = (
            self.daily_payable_outstanding_calc(self.latest_year)
        )
        self.daily_sales_outstanding_latest_year = (
            self.daily_sales_outstanding_calc(self.latest_year)
        )

        # 2. Previous Fiscal Year Calculations (T-1)
        self.daily_inventory_outstanding_previous_year = (
            self.daily_inventory_outstanding_calc(self.previous_year)
        )
        self.daily_payable_outstanding_previous_year = (
            self.daily_payable_outstanding_calc(self.previous_year)
        )
        self.daily_sales_outstanding_previous_year = (
            self.daily_sales_outstanding_calc(self.previous_year)
        )

        # 3. Third Fiscal Year Calculations (T-2)
        self.daily_inventory_outstanding_third_year = (
            self.daily_inventory_outstanding_calc(self.third_year)
        )
        self.daily_payable_outstanding_third_year = (
            self.daily_payable_outstanding_calc(self.third_year)
        )
        self.daily_sales_outstanding_third_year = (
            self.daily_sales_outstanding_calc(self.third_year)
        )

    def daily_inventory_outstanding_calc(self, year: Optional[int] = None) -> float:
        """Calculates Days Inventory Outstanding (DIO): average days to convert inventory into sales.

        Formula:
            DIO = (Average Inventory / Cost of Goods Sold) * 365

        Args:
            year (Optional[int]): Fiscal year to evaluate. Defaults to `latest_year` if None.

        Returns:
            float: Computed DIO in days, or 0.0 if COGS is zero/missing.
        """
        if year is None:
            year = self.latest_year

        cogs: float = float(self.pnl.loc[year, "Cost of Goods Sold"])
        avg_inv: float = float(self.bs.loc[year, "Average Inventory"])

        return (avg_inv / cogs) * 365.0 if cogs and cogs != 0 else 0.0

    def daily_sales_outstanding_calc(self, year: Optional[int] = None) -> float:
        """Calculates Days Sales Outstanding (DSO): average days to collect receivables.

        Formula:
            DSO = (Average Accounts Receivable / Revenue) * 365

        Args:
            year (Optional[int]): Fiscal year to evaluate. Defaults to `latest_year` if None.

        Returns:
            float: Computed DSO in days, or 0.0 if Revenue is zero/missing.
        """
        if year is None:
            year = self.latest_year

        sales: float = float(self.pnl.loc[year, "Revenue"])
        avg_ar: float = float(self.bs.loc[year, "Average Accounts Receivable"])

        return (avg_ar / sales) * 365.0 if sales and sales != 0 else 0.0

    def daily_payable_outstanding_calc(self, year: Optional[int] = None) -> float:
        """Calculates Days Payable Outstanding (DPO): average days to pay suppliers.

        Formula:
            DPO = (Average Accounts Payable / Cost of Goods Sold) * 365

        Args:
            year (Optional[int]): Fiscal year to evaluate. Defaults to `latest_year` if None.

        Returns:
            float: Computed DPO in days, or 0.0 if COGS is zero/missing.
        """
        if year is None:
            year = self.latest_year

        cogs: float = float(self.pnl.loc[year, "Cost of Goods Sold"])
        avg_ap: float = float(self.bs.loc[year, "Average Accounts Payable"])

        return (avg_ap / cogs) * 365.0 if cogs and cogs != 0 else 0.0
