import pandas as pd
from matplotlib import pyplot as plt
from analysis_model.analysis.Financial_health import FinancialHealth
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.data.Data_gathering import DataGathering
from analysis_model.analysis.Calculation_Altman_zscore import Calculation
from typing import List, Protocol, Any




class HealthFetcherProtocol(Protocol):
    """
    Protocol defining the expected interface for the financial health data provider.
    
    This interface ensures static type safety without creating tight class-coupling,
    making the Graph class easier to unit test with mock data objects.
    """
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    sector: str
    country: str
    lst_zone: List[float]
    score: float


class Graph:
    """
    Renders visual insights for Altman Z-Score financial health assessments.

    This class extracts computed financial ratios and benchmark boundaries from a data provider 
    (typically a `FinancialHealth` instance) and visualizes the risk profile using Matplotlib subplots.
    """

    def __init__(self, fetcher: Any) -> None:
        """
        Extracts raw ratios, metadata, and calculated thresholds from the fetcher object.

        Parameters
        ----------
        fetcher : HealthFetcherProtocol
            An object instance containing cleaned financial ratios, sector metadata,
            Altman Z-Score output, and zone cutoff thresholds (`lst_zone`).
        """
        # Assign Altman Z-Score individual ratio components (X1 to X5)
        self.x1: float = float(fetcher.x1)
        self.x2: float = float(fetcher.x2)
        self.x3: float = float(fetcher.x3)
        self.x4: float = float(fetcher.x4)
        self.x5: float = float(fetcher.x5)

        # Contextual Metadata
        self.sector: str = str(fetcher.sector)
        self.country: str = str(fetcher.country)

        # Base financial ratio descriptions matching X1, X2, X3, and X5 respectively
        self.ratios: List[str] = [
            "Working Capital/Total Assets",
            "Retained Earnings/Total Assets",
            "EBIT/Total Assets",
            "Sales/Total Assets",
        ]

        # Altman Z-Score decision boundaries [Distress Cutoff, Safe Cutoff]
        self.lst_zone: List[float] = fetcher.lst_zone
        self.score: float = float(fetcher.score)

        # Vectorized ratio list aligned with self.ratios for conditional filtering
        self.x_lst: List[float] = [self.x1, self.x2, self.x3, self.x5]

    def plot(self) -> plt.Figure:
        """
        Generates a 3-panel visualization dashboard for financial diagnostic data.

        Logic Overview
        --------------
        1. **Sector Handling**: Modifies `ratios` and `x_lst` dynamically. If the sector is non-manufacturing,
           `X5` (Sales/Total Assets) is stripped out, adapting to non-manufacturing Altman variants.
        2. **Visualization Breakdown**:
           - Panel 1 (`ax`): Bar chart representing standard financial ratio metrics.
           - Panel 2 (`ax1`): Distress/Grey/Safe risk zones with company's actual score plotted.
           - Panel 3 (`ax2`): Bar chart specifically isolating Market Cap to Total Liabilities (`X4`).
        3. **Dynamic Axis Padding**: Dynamically scales `ax1` Y-axis limits (`y_min`, `y_max`) 
           to keep the company score visible regardless of extreme positive/negative values.
        """
        plt.style.use("ggplot")

        # Create mutable copies to protect instance variables during dynamic manipulation
        ratios: List[str] = list(self.ratios)
        x_lst: List[float] = list(self.x_lst)

        # Dynamic adjustments: Non-manufacturing models exclude Sales/Total Assets (X5)
        if self.sector.lower() != "manufacturing":
            ratios.pop(-1)
            x_lst.pop(-1)

        fig, (ax, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

        # Panel 1: Multi-ratio breakdown
        ax.bar(ratios, x_lst, edgecolor="black", alpha=0.85)
        ax.set_title("RATIOS")
        ax.set_xlabel("Ratios")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=15)

        # Panel 2: Market valuation to liabilities ratio (X4)
        ax2.bar("Market Cap/Total Liabilities", self.x4)

        # Panel 3: Risk Zone classification chart
        ax1.scatter(
            ["Altman Z-Score"],
            [self.score],
            color="black",
            s=100,
            zorder=5,
            label="Company Score",
        )
        ax1.set_title("Z-SCORE ZONE")
        ax1.set_ylabel("Score")

        # Dynamic bound calculation to guarantee visual buffer around outliers
        y_min: float = min(0.0, self.score - 1.0)
        y_max: float = max(self.score * 1.15, self.lst_zone[1] + 1.0)

        # Shaded risk zones: Distress (< zone[0]), Grey (zone[0]-zone[1]), Safe (> zone[1])
        ax1.axhspan(
            y_min, self.lst_zone[0], color="red", alpha=0.3, label="Distress Zone "
        )
        ax1.axhspan(
            self.lst_zone[0],
            self.lst_zone[1],
            color="yellow",
            alpha=0.3,
            label="Grey Zone ",
        )
        ax1.axhspan(
            self.lst_zone[1], y_max, color="green", alpha=0.3, label="Safe Zone"
        )

        # Threshold boundary reference lines
        ax1.axhline(
            self.lst_zone[0], color="red", linestyle="--", linewidth=1.5
        )
        ax1.axhline(
            self.lst_zone[1], color="green", linestyle="--", linewidth=1.5
        )

        ax1.set_ylim(y_min, y_max)
        ax1.legend(loc="upper right")

        fig.tight_layout()
        self.fig = fig
        return fig


if __name__ == "__main__":
    data = DataGathering("AAPL")
    data.fetch_all_data()
    cleaned = StoringAndCleaning(data_fetch=data)
    cleaned.fetch_dataframe()
    cleaned.pandas_dataframe()
    calc = Calculation(fetcher=cleaned)
    calc.initializer()
    health = FinancialHealth(fetcher=calc)
    health.fetch_all_data()
    gra = Graph(fetcher=health)
    gra.plot()
