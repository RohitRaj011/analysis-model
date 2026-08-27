"""DuPont Analysis Visualization Module.

This module provides the plotting interface (`Graph`) for visual comparison of DuPont 5-Step
decomposition components across three consecutive financial years.

Data Flow:
    `ReturnOnCapital` (source) -> `Graph.__init__` (extraction) -> `Graph.plot` (Matplotlib rendering)
"""

from typing import List, Any
from matplotlib import pyplot as plt
import numpy as np

# Imports used for runtime standalone execution block (__main__)
from analysis_model.data.Data_gathering import DataGathering
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.analysis.Calculation_Dupoint import Calculation
from analysis_model.analysis.Return_on_capital import ReturnOnCapital


class Graph:
    """Renders a multi-bar chart comparing DuPont financial ratios across three fiscal years.

    Attributes:
        xaxis_value (List[str]): Metric names displayed along the X-axis.
        yaxis_value_third (List[float]): DuPont ratio values from 3 fiscal years ago.
        yaxis_value_previous (List[float]): DuPont ratio values from the previous fiscal year.
        yaxis_value_latest (List[float]): DuPont ratio values from the most recent fiscal year.
        latest_year (str): Label for the most recent fiscal year.
        previous_year (str): Label for the prior fiscal year.
        third_year (str): Label for 3 fiscal years ago.
    """

    def __init__(self, fetcher: ReturnOnCapital) -> None:
        """Extracts multi-year DuPont ratios and calendar labels from the ReturnOnCapital provider.

        Args:
            fetcher (ReturnOnCapital): Evaluated instance containing computed 5-step DuPont metrics.
        """
        # --- Fiscal Year 1 (Most Recent) Metrics ---
        self.net_profit_margin_latest_year: float = fetcher.net_profit_margin_latest_year
        self.asset_turnover_latest_year: float = fetcher.asset_turnover_latest_year
        self.equity_multiplier_latest_year: float = fetcher.equity_multiplier_latest_year
        self.tax_burden_latest_year: float = fetcher.tax_burden_latest_year
        self.interest_burden_latest_year: float = fetcher.interest_burden_latest_year
        self.dupoint_analysis_latest_year: float = fetcher.dupoint_analysis_latest_year

        # --- Fiscal Year 2 (Previous) Metrics ---
        self.net_profit_margin_previous_year: float = fetcher.net_profit_margin_previous_year
        self.asset_turnover_previous_year: float = fetcher.asset_turnover_previous_year
        self.equity_multiplier_previous_year: float = fetcher.equity_multiplier_previous_year
        self.tax_burden_previous_year: float = fetcher.tax_burden_previous_year
        self.interest_burden_previous_year: float = fetcher.interest_burden_previous_year
        self.dupoint_analysis_previous_year: float = fetcher.dupoint_analysis_previous_year

        # --- Fiscal Year 3 (Third Year Back) Metrics ---
        self.net_profit_margin_third_year: float = fetcher.net_profit_margin_third_year
        self.asset_turnover_third_year: float = fetcher.asset_turnover_third_year
        self.equity_multiplier_third_year: float = fetcher.equity_multiplier_third_year
        self.tax_burden_third_year: float = fetcher.tax_burden_third_year
        self.interest_burden_third_year: float = fetcher.interest_burden_third_year
        self.dupoint_analysis_third_year: float = fetcher.dupoint_analysis_third_year

        # Fiscal Year Header Strings
        self.latest_year: str = str(fetcher.latest_year)
        self.previous_year: str = str(fetcher.previous_year)
        self.third_year: str = str(fetcher.third_year)

        # Plot Data Vectors
        self.xaxis_value: List[str] = [
            "Net Profit Margin",
            "Asset Turnover",
            "Equity Multiplier",
            "Tax Burden",
            "Interest Burden",
            "Dupoint Score",
        ]

        self.yaxis_value_third: List[float] = [
            self.net_profit_margin_third_year,
            self.asset_turnover_third_year,
            self.equity_multiplier_third_year,
            self.tax_burden_third_year,
            self.interest_burden_third_year,
            self.dupoint_analysis_third_year,
        ]

        self.yaxis_value_previous: List[float] = [
            self.net_profit_margin_previous_year,
            self.asset_turnover_previous_year,
            self.equity_multiplier_previous_year,
            self.tax_burden_previous_year,
            self.interest_burden_previous_year,
            self.dupoint_analysis_previous_year,
        ]

        self.yaxis_value_latest: List[float] = [
            self.net_profit_margin_latest_year,
            self.asset_turnover_latest_year,
            self.equity_multiplier_latest_year,
            self.tax_burden_latest_year,
            self.interest_burden_latest_year,
            self.dupoint_analysis_latest_year,
        ]

    def plot(self) -> plt.Figure:
        """Constructs a side-by-side 3-year grouped bar chart.

        Design Rationale:
            Grouped bar offsets (`x - width`, `x`, `x + width`) are utilized to cleanly render
            year-over-year shifts across heterogeneous financial metrics without visual overlap.
        """
        plt.style.use("ggplot")
        fig, ax = plt.subplots()

        # Generate discrete numerical indices for category positioning
        x: np.ndarray = np.arange(len(self.xaxis_value))
        width: float = 0.25  # Bar width offset factor

        # Render offset bars for each fiscal year
        ax.bar(x - width, self.yaxis_value_third, width, label=self.third_year)
        ax.bar(x, self.yaxis_value_previous, width, label=self.previous_year)
        ax.bar(x + width, self.yaxis_value_latest, width, label=self.latest_year)

        # Formatting & Chart Annotations
        ax.set_title("DUPOINT ANALYSIS")
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(self.xaxis_value, rotation=15)
        ax.legend()

        fig.tight_layout()
        self.fig = fig
        return fig


if __name__ == "__main__":
    # Integration smoke test: Ingest, clean, compute, and plot AAPL data
    data: DataGathering = DataGathering("AAPL")
    data.fetch_all_data()

    storage: StoringAndCleaning = StoringAndCleaning(data_fetch=data)
    storage.fetch_dataframe()

    calc: Calculation = Calculation(fetcher=storage)
    calc.master_initializer()

    roc: ReturnOnCapital = ReturnOnCapital(fetcher=calc)
    roc.calc()

    gra: Graph = Graph(fetcher=roc)
    gra.plot()
