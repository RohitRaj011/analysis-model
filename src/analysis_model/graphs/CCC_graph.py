"""Cash Conversion Cycle (CCC) Waterfall Graph Visualization Module.

This module provides the `Graph` plotting class to render multi-year waterfall bar charts
comparing Days Inventory Outstanding (DIO), Days Sales Outstanding (DSO), Days Payable
Outstanding (DPO), and total Cash Conversion Cycle (CCC).

Data Flow:
    `OperationEFFICIENCY` -> `Graph.__init__` (Extraction) -> `Graph.plot` (Waterfall Rendering)
"""

from typing import List, Tuple, Any
import matplotlib.pyplot as plt

# Imports for standalone integration smoke test (__main__)
from analysis_model.data.Data_gathering import DataGathering
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.analysis.Calculations_CCC import Calculation
from analysis_model.analysis.operational_efficency import OperationEFFICIENCY


class Graph:
    """Renders a 3-year stacked waterfall chart visualizing working capital components.

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
        ccc_latest_year (float): Net Cash Conversion Cycle days for year T.
        ccc_previous_year (float): Net Cash Conversion Cycle days for year T-1.
        ccc_third_year (float): Net Cash Conversion Cycle days for year T-2.
        latest_year (int): Fiscal year T label.
        previous_year (int): Fiscal year T-1 label.
        third_year (int): Fiscal year T-2 label.
        xaxis (List[str]): X-axis metric category labels.
    """

    def __init__(self, fetcher: OperationEFFICIENCY) -> None:
        """Extracts component days and net CCC calculations from OperationEFFICIENCY provider.

        Args:
            fetcher (OperationEFFICIENCY): Evaluated instance holding DIO, DSO, DPO, and CCC.
        """
        # --- Year T Metrics ---
        self.daily_inventory_outstanding_latest_year: float = (
            fetcher.daily_inventory_outstanding_latest_year
        )
        self.daily_sales_outstanding_latest_year: float = (
            fetcher.daily_sales_outstanding_latest_year
        )
        self.daily_payable_outstanding_latest_year: float = (
            fetcher.daily_payable_outstanding_latest_year
        )
        self.ccc_latest_year: float = fetcher.ccc_latest_year

        # --- Year T-1 Metrics ---
        self.daily_inventory_outstanding_previous_year: float = (
            fetcher.daily_inventory_outstanding_previous_year
        )
        self.daily_sales_outstanding_previous_year: float = (
            fetcher.daily_sales_outstanding_previous_year
        )
        self.daily_payable_outstanding_previous_year: float = (
            fetcher.daily_payable_outstanding_previous_year
        )
        self.ccc_previous_year: float = fetcher.ccc_previous_year

        # --- Year T-2 Metrics ---
        self.daily_inventory_outstanding_third_year: float = (
            fetcher.daily_inventory_outstanding_third_year
        )
        self.daily_sales_outstanding_third_year: float = (
            fetcher.daily_sales_outstanding_third_year
        )
        self.daily_payable_outstanding_third_year: float = (
            fetcher.daily_payable_outstanding_third_year
        )
        self.ccc_third_year: float = fetcher.ccc_third_year

        # Fiscal Year Labels
        self.latest_year: int = fetcher.latest_year
        self.previous_year: int = fetcher.previous_year
        self.third_year: int = fetcher.third_year

        self.xaxis: List[str] = [
            "Inventory Outstanding",
            "Sales Outstanding",
            "Payable Outstanding",
            "Cash Conversion Cycle",
        ]

    def plot(self) -> plt.Figure:
        """Generates a 3-panel stacked waterfall chart showing CCC mechanics over time.

        Design Rationale:
            Waterfall steps use floating baseline origins (`bottom=starts`) so each working 
            capital component accumulates visually (DIO + DSO - DPO = Net CCC). Dotted connector 
            lines highlight step transitions between working capital stages.
        """
        plt.style.use("ggplot")
        fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

        # Color scheme setup
        pos_color: str = "#2ecc71"   # Green: Increases asset holding / positive impact
        neg_color: str = "#e74c3c"   # Red: Payables reduction / outflow impact
        total_color: str = "#34495e"  # Dark Slate: Final net CCC metric

        # Data structure: [(starts, heights, raw_labels, dio, dso, dpo, chart_title)]
        datasets: List[Tuple[List[float], List[float], List[float], float, float, float, str]] = [
            (
                # Starts (baselines)
                [
                    0,
                    self.daily_inventory_outstanding_latest_year,
                    self.daily_inventory_outstanding_latest_year
                    + self.daily_sales_outstanding_latest_year,
                    0,
                ],
                # Heights (bar lengths)
                [
                    self.daily_inventory_outstanding_latest_year,
                    self.daily_sales_outstanding_latest_year,
                    -self.daily_payable_outstanding_latest_year,
                    self.ccc_latest_year,
                ],
                # Value labels
                [
                    self.daily_inventory_outstanding_latest_year,
                    self.daily_sales_outstanding_latest_year,
                    -self.daily_payable_outstanding_latest_year,
                    self.ccc_latest_year,
                ],
                self.daily_inventory_outstanding_latest_year,
                self.daily_sales_outstanding_latest_year,
                self.daily_payable_outstanding_latest_year,
                f"Cash Conversion Cycle - {self.latest_year}",
            ),
            (
                [
                    0,
                    self.daily_inventory_outstanding_previous_year,
                    self.daily_inventory_outstanding_previous_year
                    + self.daily_sales_outstanding_previous_year,
                    0,
                ],
                [
                    self.daily_inventory_outstanding_previous_year,
                    self.daily_sales_outstanding_previous_year,
                    -self.daily_payable_outstanding_previous_year,
                    self.ccc_previous_year,
                ],
                [
                    self.daily_inventory_outstanding_previous_year,
                    self.daily_sales_outstanding_previous_year,
                    -self.daily_payable_outstanding_previous_year,
                    self.ccc_previous_year,
                ],
                self.daily_inventory_outstanding_previous_year,
                self.daily_sales_outstanding_previous_year,
                self.daily_payable_outstanding_previous_year,
                f"Cash Conversion Cycle - {self.previous_year}",
            ),
            (
                [
                    0,
                    self.daily_inventory_outstanding_third_year,
                    self.daily_inventory_outstanding_third_year
                    + self.daily_sales_outstanding_third_year,
                    0,
                ],
                [
                    self.daily_inventory_outstanding_third_year,
                    self.daily_sales_outstanding_third_year,
                    -self.daily_payable_outstanding_third_year,
                    self.ccc_third_year,
                ],
                [
                    self.daily_inventory_outstanding_third_year,
                    self.daily_sales_outstanding_third_year,
                    -self.daily_payable_outstanding_third_year,
                    self.ccc_third_year,
                ],
                self.daily_inventory_outstanding_third_year,
                self.daily_sales_outstanding_third_year,
                self.daily_payable_outstanding_third_year,
                f"Cash Conversion Cycle - {self.third_year}",
            ),
        ]

        # Iterate through subplots and datasets concurrently
        for ax, (starts, heights, labels, dio, dso, dpo, title) in zip(axes, datasets):
            bar_colors: List[str] = [
                pos_color if dio >= 0 else neg_color,
                pos_color if dso >= 0 else neg_color,
                neg_color if dpo >= 0 else pos_color,
                total_color,
            ]

            ax.bar(
                self.xaxis,
                heights,
                bottom=starts,
                color=bar_colors,
                width=0.45,
                edgecolor="none",
            )

            ax.axhline(0, color="black", linewidth=1.2, zorder=3)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Draw waterfall step connector lines between consecutive component bars
            for i in range(len(heights) - 2):
                end_val: float = starts[i] + heights[i]
                ax.plot(
                    [i - 0.225, i + 1 + 0.225],
                    [end_val, end_val],
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                )

            # Annotate numerical text callouts on top/bottom of each bar
            for start, height, label_val, x_label in zip(starts, heights, labels, self.xaxis):
                y_pos: float = start + height + (1.5 if height >= 0 else -3.5)
                sign: str = "+" if label_val > 0 else ""

                ax.text(
                    x_label,
                    y_pos,
                    f"{sign}{label_val:.1f} days",
                    va="bottom" if height >= 0 else "top",
                    ha="center",
                    fontweight="bold",
                    color="black",
                    fontsize=9,
                )

            # Calculate dynamic Y-axis bounds with padding
            min_y: float = min(0, min(starts), min(
                s + h for s, h in zip(starts, heights)))
            max_y: float = max(0, max(starts), max(
                s + h for s, h in zip(starts, heights)))
            padding: float = (max_y - min_y) * 0.15
            ax.set_ylim(min_y - padding, max_y + padding)

            ax.tick_params(axis="x", pad=8)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
            ax.set_ylabel("Days", fontsize=10)

            # Clean borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        axes[-1].set_xlabel("Ratios", fontsize=11)

        fig.tight_layout()
        self.fig = fig
        return fig


if __name__ == "__main__":
    # Integration smoke test: Run pipeline for AAPL
    data_fetcher: DataGathering = DataGathering("AAPL")
    data_fetcher.fetch_all_data()

    cleaned: StoringAndCleaning = StoringAndCleaning(data_fetch=data_fetcher)
    cleaned.fetch_dataframe()
    cleaned.pandas_dataframe()

    cal: Calculation = Calculation(fetcher=cleaned)
    cal.master_initializer()

    operation: OperationEFFICIENCY = OperationEFFICIENCY(fetcher=cal)
    operation.calc()

    graph: Graph = Graph(fetcher=operation)
    graph.plot()
