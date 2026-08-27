import pandas as pd
from typing import List, Optional, Set, Protocol, Any


class CalculationFetcherProtocol(Protocol):
    """
    Protocol defining the expected interface for the ratio calculation provider.
    
    Ensures static type checking without creating direct class coupling between
    the Calculation module and FinancialHealth.
    """
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    sector: str
    country: str


class FinancialHealth:
    """
    Evaluates financial health by applying specific Altman Z-Score weighting formulas 
    and classifying the target company into risk zones.

    Attributes
    ----------
    x1 : float
        Working Capital / Total Assets ratio.
    x2 : float
        Retained Earnings / Total Assets ratio.
    x3 : float
        EBIT / Total Assets ratio.
    x4 : float
        Market Value of Equity / Total Liabilities ratio.
    x5 : float
        Sales / Total Assets ratio.
    sector : str
        Normalized (lowercase) sector of the target entity.
    country : str
        Normalized (lowercase) country of origin.
    score : Optional[float]
        The calculated Altman Z-Score value.
    lst_zone : List[float]
        Threshold boundaries [Safe Cutoff, Distress Cutoff] corresponding to the model variant.
    """

    # Sectors using the standard manufacturing/capital-intensive weighting formula
    MANUFACTURING_SECTORS: Set[str] = {
        "industrials",
        "basic-materials",
        "consumer-cyclical",
        "consumer-defensive",
        "healthcare",
    }

    def __init__(self, fetcher: Any) -> None:
        """
        Initializes ratio variables and metadata from the calculation engine output.

        Parameters
        ----------
        fetcher : CalculationFetcherProtocol
            An object instance containing calculated ratio components (x1 through x5),
            as well as entity sector and country metadata.
        """
        self.x1: float = float(fetcher.x1)
        self.x2: float = float(fetcher.x2)
        self.x3: float = float(fetcher.x3)
        self.x4: float = float(fetcher.x4)
        self.x5: float = float(fetcher.x5)

        self.sector: str = str(fetcher.sector).lower().strip()
        self.country: str = str(fetcher.country).lower().strip()

        self.score: Optional[float] = None
        self.lst_zone: List[float] = []

    def fetch_all_data(self) -> None:
        """
        Executes the full evaluation pipeline by computing the Z-Score first
        and determining distress zone boundaries second.
        """
        self.score_calc()
        self.zone()

    def score_calc(self) -> float:
        """
        Calculates the appropriate Altman Z-Score based on entity region and sector.

        Formula Selection Rationale
        ---------------------------
        1. **Manufacturing / Capital-Intensive Sectors (Non-US)**:
           Uses the classic 5-factor model including asset turnover ($X_5$).
           `Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5`

        2. **Non-Manufacturing / Service Sectors (Non-US)**:
           Uses the 4-factor emerging markets / non-manufacturing variant ($Z''$) 
           to remove industry turnover bias.
           `Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4`

        3. **US Market Variant**:
           Applies the 4-factor non-manufacturing formula with a `+ 3.25` constant baseline.

        Returns
        -------
        float
            The final calculated Z-Score assigned to `self.score`.
        """
        if self.country != "usa":
            if self.sector in self.MANUFACTURING_SECTORS:
                self.score = (
                    1.2 * self.x1
                    + 1.4 * self.x2
                    + 3.3 * self.x3
                    + 0.6 * self.x4
                    + self.x5
                )
            else:
                self.score = (
                    6.56 * self.x1
                    + 3.26 * self.x2
                    + 6.72 * self.x3
                    + 1.05 * self.x4
                )
        else:
            self.score = (
                3.25
                + 6.56 * self.x1
                + 3.26 * self.x2
                + 6.72 * self.x3
                + 1.05 * self.x4
            )

        return self.score

    def zone(self) -> List[float]:
        """
        Sets the decision boundary cutoffs used downstream for plotting or classification.

        Returns
        -------
        List[float]
            A two-element list representing `[upper_threshold, lower_threshold]`.
            - Manufacturing sectors: `[2.9, 1.23]`
            - Non-manufacturing sectors: `[2.6, 1.1]`
        """
        if self.sector in self.MANUFACTURING_SECTORS:
            self.lst_zone = [2.9, 1.23]
        else:
            self.lst_zone = [2.6, 1.1]

        return self.lst_zone
