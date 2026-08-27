"""Operational Efficiency & Working Capital Synthesis Module.

This module consumes component day metrics (DIO, DSO, DPO) derived from historical financial statements
and calculates the net Cash Conversion Cycle (CCC) across three consecutive fiscal years.

Financial Math Reference:
    Net CCC = Days Inventory Outstanding (DIO) + Days Sales Outstanding (DSO) - Days Payable Outstanding (DPO)

Interpretation:
    - Measures the total time (in days) required for a company to convert its investments in inventory
      and other operational resources into cash flows from sales.
    - A lower or negative CCC indicates superior working capital efficiency (e.g., funding operations
      using supplier credit).
"""

from typing import Optional

# Forward reference type hint import
from analysis_model.analysis.Calculations_CCC import Calculation as CCCCalculation


class OperationEFFICIENCY:
    """Synthesizes DIO, DSO, and DPO component metrics to calculate multi-year net Cash Conversion Cycles.

    Attributes:
        daily_inventory_outstanding_latest_year (float): DIO for year T.
        daily_sales_outstanding_latest_year (float): DSO for year T.
        daily_payable_outstanding_latest_year (float): DPO for year T.
        daily_inventory_outstanding_previous_year (float): DIO for year T-1.
        daily_sales_outstanding_previous_year (float): DSO for year T-1.
        daily_payable_outstanding_previous_year (float): DPO for year T-1.
        daily_inventory_outstanding_third_year (float): DIO for year T-2.
        daily_sales_outstanding_third_year (float): DSO for year T-2.
        daily_payable_outstanding_third_year (float): DPO for year T-2.
        ccc_latest_year (Optional[float]): Computed Net CCC for year T.
        ccc_previous_year (Optional[float]): Computed Net CCC for year T-1.
        ccc_third_year (Optional[float]): Computed Net CCC for year T-2.
        latest_year (int): Year T fiscal label.
        previous_year (int): Year T-1 fiscal label.
        third_year (int): Year T-2 fiscal label.
    """

    def __init__(self, fetcher: CCCCalculation) -> None:
        """Extracts component days and fiscal year metadata from the calculation layer.

        Args:
            fetcher (CCCCalculation): Evaluated instance holding calculated DIO, DSO, and DPO metrics.
        """
        # --- Fiscal Year T (Latest Year) Metrics ---
        self.daily_inventory_outstanding_latest_year: float = float(
            fetcher.daily_inventory_outstanding_latest_year or 0.0
        )
        self.daily_sales_outstanding_latest_year: float = float(
            fetcher.daily_sales_outstanding_latest_year or 0.0
        )
        self.daily_payable_outstanding_latest_year: float = float(
            fetcher.daily_payable_outstanding_latest_year or 0.0
        )

        # --- Fiscal Year T-1 (Previous Year) Metrics ---
        self.daily_inventory_outstanding_previous_year: float = float(
            fetcher.daily_inventory_outstanding_previous_year or 0.0
        )
        self.daily_sales_outstanding_previous_year: float = float(
            fetcher.daily_sales_outstanding_previous_year or 0.0
        )
        self.daily_payable_outstanding_previous_year: float = float(
            fetcher.daily_payable_outstanding_previous_year or 0.0
        )

        # --- Fiscal Year T-2 (Third Year) Metrics ---
        self.daily_inventory_outstanding_third_year: float = float(
            fetcher.daily_inventory_outstanding_third_year or 0.0
        )
        self.daily_sales_outstanding_third_year: float = float(
            fetcher.daily_sales_outstanding_third_year or 0.0
        )
        self.daily_payable_outstanding_third_year: float = float(
            fetcher.daily_payable_outstanding_third_year or 0.0
        )

        # CCC Placeholders
        self.ccc_latest_year: Optional[float] = None
        self.ccc_previous_year: Optional[float] = None
        self.ccc_third_year: Optional[float] = None

        # Year Labels
        self.latest_year: int = int(fetcher.latest_year)
        self.previous_year: int = int(fetcher.previous_year)
        self.third_year: int = int(fetcher.third_year)

    def calc(self) -> None:
        """Triggers Net Cash Conversion Cycle calculations across all three target fiscal years."""
        self.ccc_latest_year = self.cash_conversion_cycle(
            inv=self.daily_inventory_outstanding_latest_year,
            sales=self.daily_sales_outstanding_latest_year,
            payable=self.daily_payable_outstanding_latest_year,
        )

        self.ccc_previous_year = self.cash_conversion_cycle(
            inv=self.daily_inventory_outstanding_previous_year,
            sales=self.daily_sales_outstanding_previous_year,
            payable=self.daily_payable_outstanding_previous_year,
        )

        self.ccc_third_year = self.cash_conversion_cycle(
            inv=self.daily_inventory_outstanding_third_year,
            sales=self.daily_sales_outstanding_third_year,
            payable=self.daily_payable_outstanding_third_year,
        )

    @staticmethod
    def cash_conversion_cycle(inv: float, sales: float, payable: float) -> float:
        """Calculates Net Cash Conversion Cycle (CCC) in days.

        Formula:
            CCC = DIO + DSO - DPO

        Args:
            inv (float): Days Inventory Outstanding (DIO).
            sales (float): Days Sales Outstanding (DSO).
            payable (float): Days Payable Outstanding (DPO).

        Returns:
            float: Net Cash Conversion Cycle in days.
        """
        return (inv + sales) - payable
