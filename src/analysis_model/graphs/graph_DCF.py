"""Discounted Cash Flow (DCF) Visualization Dashboard Module.

This module provides the `Graph` plotting class to render a 3-panel dashboard:
    1. Historical vs. Projected Free Cash Flow (FCF) side-by-side bar chart.
    2. Equity Value Bridge waterfall chart (Enterprise Value + Cash - Debt = Equity Value).
    3. Current Market Price vs. DCF Intrinsic Per-Share Target valuation comparison.

Data Flow:
    `Calculation` & `Valuation` -> `Graph.__init__` (Extraction) -> `Graph.plot` (Dashboard Rendering)
"""

from typing import List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imports for standalone integration smoke test (__main__)
from analysis_model.data.Data_gathering import DataGathering
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.analysis.Calculations_DCF import Calculation
from analysis_model.analysis.valuation import Valuation


class Graph:
    """Renders a 3-subplot dashboard illustrating DCF cash flows, valuation bridge, and per-share targets.

    Attributes:
        past_years (List[int]): Historical fiscal years.
        latest_year (int): Most recent historical fiscal year.
        free_cashflow (pd.Series): Historical free cash flow series.
        projected_years (List[int]): Projected future fiscal years.
        projected_cashflow (pd.Series): Forecasted free cash flow projections.
        ev (float): Enterprise Value computed by the DCF engine.
        cash (float): Cash & cash equivalents balance.
        debt (float): Total short-term and long-term debt liabilities.
        equity_value (float): Implied net Equity Value (EV + Cash - Debt).
        current_price (float): Real-time market trading price per share.
        intrinsic_value (float): Implied DCF intrinsic value target per share.
    """

    def __init__(self, fetcher: Calculation, val_fetcher: Valuation) -> None:
        """Extracts projected cash flows, enterprise bridges, and per-share metrics.

        Args:
            fetcher (Calculation): Calculation engine holding historical data & growth forecasts.
            val_fetcher (Valuation): Valuation evaluator holding enterprise adjustments and targets.
        """
        # --- Data Extraction from Calculation ---
        self.past_years: List[int] = [int(y) for y in list(fetcher.index)]
        self.latest_year: int = int(fetcher.latest_year)
        self.free_cashflow: pd.Series = fetcher.data["Free cashflow"]

        # Shift projection years forward from the last historical year index
        self.projected_years: List[int] = [
            self.latest_year + int(i) for i in fetcher.projections.index
        ]
        self.projected_cashflow: pd.Series = fetcher.projections["Projected Cashflow"]

        # --- Data Extraction from Valuation ---
        self.ev: float = float(val_fetcher.ev)
        self.cash: float = float(val_fetcher.cash_and_cash_equivalent)
        self.debt: float = float(val_fetcher.total_debts)
        self.equity_value: float = float(val_fetcher.equity_value)

        self.current_price: float = float(val_fetcher.current_price)
        self.intrinsic_value: float = float(val_fetcher.intrisnic_value)

    def plot(self) -> plt.Figure:
        """Constructs a 3-panel valuation dashboard.

        Design Rationale:
            - Panel 1: Compares historical cash flow trends against multi-period growth projections.
            - Panel 2: Employs a floating baseline waterfall chart (`bottom=starts`) to illustrate
              how enterprise value transitions to equity value.
            - Panel 3: Color-codes intrinsic value (Green if intrinsic > market price, Red if overvalued).
        """
        plt.style.use("ggplot")
        fig, axes = plt.subplots(3, 1, figsize=(6, 18))

        # ----------------------------------------------------
        # Subplot 1: Historical vs Projected Free Cash Flow
        # ----------------------------------------------------
        ax1: plt.Axes = axes[0]
        ax1.bar(
            self.past_years,
            self.free_cashflow,
            color="#3498db",
            label="Historical FCF",
        )
        ax1.bar(
            self.projected_years,
            self.projected_cashflow,
            color="#f39c12",
            label="Projected FCF",
        )
        ax1.set_title("Historical vs Projected Free Cashflow")
        ax1.set_ylabel("Cashflow ($)")
        ax1.set_xticks(self.past_years + self.projected_years)
        ax1.legend()

        # ----------------------------------------------------
        # Subplot 2: Equity Value Bridge (Waterfall Chart Logic)
        # ----------------------------------------------------
        ax2: plt.Axes = axes[1]
        bridge_labels: List[str] = [
            "Enterprise Value",
            "Cash",
            "Debt",
            "Equity Value",
        ]

        # Floating bar baselines and displacements: EV + Cash - Debt = Equity Value
        starts: List[float] = [0.0, self.ev, self.ev + self.cash, 0.0]
        heights: List[float] = [self.ev, self.cash, -self.debt, self.equity_value]
        colors: List[str] = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

        ax2.bar(bridge_labels, heights, bottom=starts, color=colors)
        ax2.axhline(0, color="black", linewidth=1)

        # Label each bar segment in the vertical center
        for i, h in enumerate(heights):
            ax2.text(
                i,
                starts[i] + h / 2.0,
                f"{h:,.0f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )
        ax2.set_title("Equity Value Bridge")
        ax2.set_ylabel("Value ($)")

        # ----------------------------------------------------
        # Subplot 3: Current Price vs Intrinsic Value Comparison
        # ----------------------------------------------------
        ax3: plt.Axes = axes[2]
        labels: List[str] = ["Current Price", "Intrinsic Value"]
        values: List[float] = [self.current_price, self.intrinsic_value]

        # Highlight green if undervalued (upside potential), red if overvalued
        bar_colors: List[str] = [
            "#7f8c8d",
            "#27ae60" if self.intrinsic_value > self.current_price else "#c0392b",
        ]
        bars = ax3.bar(labels, values, color=bar_colors, width=0.4)
        ax3.set_title("Current Price vs Intrinsic Value")
        ax3.set_ylabel("Price per Share ($)")

        # Annotate numerical price tags above bars
        for bar in bars:
            yval: float = float(bar.get_height())
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + (yval * 0.02),
                f"${yval:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.tight_layout()
        self.fig = fig
        return fig


if __name__ == "__main__":
    # Integration smoke test: Run end-to-end DCF pipeline for AAPL and display charts
    data_fetcher: DataGathering = DataGathering("AAPL")
    
    cleaned: StoringAndCleaning = StoringAndCleaning(data_fetch=data_fetcher)
    cleaned.fetch_dataframe()
    cleaned.pandas_dataframe()

    cal: Calculation = Calculation(fetcher=cleaned)
    cal.fetch_data()

    val: Valuation = Valuation(fetcher=cal)
    val.calc()

    gra: Graph = Graph(fetcher=cal, val_fetcher=val)
    gra.plot()
