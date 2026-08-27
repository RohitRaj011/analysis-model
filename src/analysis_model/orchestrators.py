from typing import Optional, Dict, Any, Union
from analysis_model.data.Data_gathering import DataGathering
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.analysis.Calculations_DCF import Calculation
from analysis_model.analysis.valuation import Valuation
from analysis_model.analysis import Calculation_Dupoint
from analysis_model.analysis.Return_on_capital import ReturnOnCapital
from analysis_model.analysis import Calculation_Altman_zscore
from analysis_model.analysis.Financial_health import FinancialHealth
from analysis_model.analysis import Calculations_CCC
from analysis_model.analysis.operational_efficency import OperationEFFICIENCY
from analysis_model.data.cached_fetch import load_cached_payload, load_or_fetch_gatherer
from analysis_model.graphs import CCC_graph
from analysis_model.graphs import graph_zscore
from analysis_model.graphs import graph_DCF
from analysis_model.graphs import graph_dupoint_analysispy
import json
from datetime import date
import os


def _project_root() -> str:
    """Repo root (two levels above this package file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


MAX_DAILY_RUNS: int = 30


def tracker_status(max_calls: int = MAX_DAILY_RUNS) -> Dict[str, Any]:
    """Read today's analysis-run count without incrementing.

    One tracker unit = one uncached analysis run (about six FMP HTTP calls).
    Cached re-runs for the same ticker/day do not increment the counter.
    """
    file_path: str = os.path.join(_project_root(), "tracker.json")
    today_str: str = str(date.today())
    calls = 0
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                saved_data: Dict[str, Any] = json.load(file)
                if saved_data.get("date") == today_str:
                    calls = int(saved_data.get("calls", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            calls = 0
    remaining = max(0, max_calls - calls)
    return {
        "date": today_str,
        "calls": calls,
        "max_calls": max_calls,
        "remaining": remaining,
    }


def api_tracker(max_calls: int = MAX_DAILY_RUNS) -> bool:
    """Tracks daily uncached analysis runs against `tracker.json`.

    One increment = one uncached analysis run (about six FMP HTTP calls), not each endpoint.
    Cached re-runs for the same ticker on the same calendar day do not increment.

    Args:
        max_calls (int): Maximum uncached analysis runs per calendar day. Defaults to 30.

    Returns:
        bool: True if the request quota for the current day has not been exceeded;
              False if the rate limit has been hit or exceeded.
    """
    file_path: str = os.path.join(_project_root(), "tracker.json")
    today_str: str = str(date.today())

    data: Dict[str, Union[str, int]] = {"date": today_str, "calls": 0}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                saved_data: Dict[str, Any] = json.load(file)
                if saved_data.get("date") == today_str:
                    data["calls"] = saved_data.get("calls", 0)
        except (json.JSONDecodeError, OSError):
            pass

    if data["calls"] >= max_calls:
        print(
            f" Daily API call limit ({max_calls}) reached for today ({today_str}).")
        return False

    data["calls"] += 1
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"API Call allowed: {data['calls']}/{max_calls} used today.")
    return True


class GatheringAndStoringAndCleaning():
    """Initial ETL (Extract, Transform, Load) pipeline stage for financial stock analysis.

    This class encapsulates data ingestion from remote financial APIs and coordinates 
    the transformation of unstructured JSON or raw response payloads into clean, 
    tabular `pandas.DataFrame` structures. This clean object acts as the primary 
    upstream data supplier injected into subsequent analysis pipelines.

    Attributes:
        gather (Optional[DataGathering]): Instance handling the raw HTTP network interactions.
        storing (Optional[StoringAndCleaning]): Instance executing data-cleaning routines and storing DataFrames.
        ticker (str): The equity ticker symbol targeted for extraction (e.g., 'AAPL').
        api_limit (bool): Internal tracker state recording whether the rate limit check passed.
    """

    def __init__(self, ticker: str) -> None:
        """Initializes the ETL controller with a specific stock ticker.

        Args:
            ticker (str): The market ticker symbol to extract financial statements for.
        """
        self.gather: Optional[DataGathering] = None
        self.storing: Optional[StoringAndCleaning] = None
        self.ticker: str = ticker
        self.api_limit: bool = True
        self.from_cache: bool = False
        self.premium_warning: Optional[str] = None

    def initialize_data(self) -> None:
        """Triggers the raw network extraction layer after validating rate-limit quotas.

        Checks `api_tracker()` first. If execution is permitted, instantiates `DataGathering` 
        and pulls financial statements across balance sheets, income statements, and cash flows.
        """
        if load_cached_payload(self.ticker):
            self.gather, self.from_cache = load_or_fetch_gatherer(self.ticker)
            self.api_limit = True
            print("Using cached statements for today; tracker not incremented.")
            return

        if not api_tracker():
            print("Aborting data fetch due to daily rate limit.")
            self.api_limit = False
            return
        self.api_limit = True
        print("Gathering raw data...")
        self.gather, self.from_cache = load_or_fetch_gatherer(self.ticker)

    def process_data(self) -> None:
        """Executes parsing, normalization, and DataFrame compilation.

        Transforms raw network outputs into structured Pandas DataFrames held within 
        the `self.storing` object. Ensures downstream models receive standard tabular schema.
        """
        """Step 2: Restructures raw network statements into clean Pandas DataFrames.

        Note: self.storing.pandas_dataframe() is safely called here, ensuring that 
        tabular structures compile immediately before calculation routines trigger.
        """
        print("Storing and processing data...")
        self.storing = StoringAndCleaning(
            data_fetch=self.gather)
        self.storing.fetch_dataframe()
        self.premium_warning = getattr(self.storing, "premium_warning", None)


class DiscountedCashFlow():
    """Orchestration pipeline for multi-period Discounted Cash Flow (DCF) intrinsic valuation.

    Consumes cleaned financial data (`StoringAndCleaning`) to model future Free Cash Flows (FCF),
    applies Weighted Average Cost of Capital (WACC) discount rates, calculates terminal enterprise value, 
    and converts the result into per-share intrinsic valuation verdicts.

    Attributes:
        cal (Optional[Calculation]): Financial math engine performing multi-period cash flow projections.
        val (Optional[Valuation]): Shares-outstanding and equity bridge evaluator for target price logic.
        graph (Optional[graph_DCF.Graph]): Matplotlib rendering engine plotting DCF trajectories.
        fetcher (StoringAndCleaning): Injected data container providing cleaned financial DataFrames.
        main_value (Optional[str]): Human-readable textual summary of the valuation verdict.
    """

    def __init__(
        self,
        data: StoringAndCleaning,
        const_growth_rate: float = 0.02,
        projection_years: int = 5,
    ) -> None:
        """Injects the cleaned data dependency into the DCF pipeline.

        Args:
            data (StoringAndCleaning): The initialized data layer holding target DataFrames.
            const_growth_rate: Perpetual growth rate (default 2%).
            projection_years: Discrete forecast horizon (default 5).
        """
        # Step 3: Inject the clean parser instance into the core calculator engine
        self.cal: Optional[Calculation] = None

        # Step 4: Inject the calculations engine into the per-share valuation evaluator
        self.val: Optional[Valuation] = None

        # Step 5: Inject the valuation instance into your charting wrapper
        self.graph: Optional[graph_DCF.Graph] = None

        self.fetcher: StoringAndCleaning = data
        self.const_growth_rate: float = const_growth_rate
        self.projection_years: int = projection_years

        # Ultimate container holding the terminal text presentation status status string
        self.main_value: Optional[str] = None

    def run_calculations(self) -> None:
        """Executes the mathematical calculations for DCF (e.g., FCF growth rates, discounting)."""
        print("Running calculations...")
        self.cal = Calculation(
            fetcher=self.fetcher,
            const_growth_rate=self.const_growth_rate,
            projection_years=self.projection_years,
        )
        self.cal.fetch_data()

    def evaluate_valuation(self) -> Optional[str]:
        """Calculates per-share intrinsic value and determines stock pricing verdicts.

        Returns:
            Optional[str]: Summary string indicating whether the stock is undervalued/overvalued.
        """
        print("Calculating valuation...")
        self.val = Valuation(fetcher=self.cal)
        self.val.calc()
        # Captures the final property generated by your Valuation script
        self.main_value = self.val.value
        return self.main_value

    def generate_plots(self) -> None:
        """Constructs visual graphs of projected vs. discounted cash flows."""
        print("Generating Discounted Cashflow Graph...")
        self.graph = graph_DCF.Graph(
            fetcher=self.cal, val_fetcher=self.val)
        self.graph.plot()

    def master_initialization(self) -> None:
        """Executes the DCF pipeline in strict sequential dependency order."""
        self.run_calculations()
        self.evaluate_valuation()
        self.generate_plots()
        print("DCF Pipeline execution complete.")


class Dupoint():
    """Orchestration engine for DuPont 3-Step / 5-Step Financial Decomposition analysis.

    Breaks Return on Equity (ROE) into component drivers: Profit Margin, Asset Turnover, 
    and Financial Leverage. This helps isolate whether profitability is driven by operational 
    efficiency, asset utilization, or financial leverage.

    Attributes:
        calculation (Calculation_Dupoint.Calculation): Ratios math module.
        return_on_capital (Optional[ReturnOnCapital]): Capital return metrics evaluator.
        final_value (Optional[Any]): Container for terminal DuPont values.
        graph (Optional[graph_dupoint_analysispy.Graph]): Visualizer for DuPont breakdown trends.
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Injects clean DataFrame sources into the DuPont module.

        Args:
            fetcher (StoringAndCleaning): Cleaned financial DataFrame instance.
        """
        self.calculation: Calculation_Dupoint.Calculation = Calculation_Dupoint.Calculation(
            fetcher=fetcher)
        self.return_on_capital: Optional[ReturnOnCapital] = None
        self.final_value: Optional[Any] = None
        self.graph: Optional[graph_dupoint_analysispy.Graph] = None

    def master_initializer(self) -> None:
        """Orchestrates sequential DuPont ratio calculations, return evaluation, and graphing."""
        self.calculation_initializer()
        self.calc_return_on_capital()
        self.generate_plots()
        print("Dupoint Pipeline execution complete.")

    def calculation_initializer(self) -> None:
        """Triggers underlying metric computation for margins, turnovers, and leverage ratios."""
        print("Running Dupoint Calculations.......")
        self.calculation.master_initializer()

    def calc_return_on_capital(self) -> None:
        """Computes ROIC/ROE synthesis using calculated DuPont metrics."""
        print("Calculation Return on Capital.......")
        self.return_on_capital = ReturnOnCapital(fetcher=self.calculation)
        self.return_on_capital.calc()
        self.final_value = self.return_on_capital.dupoint_analysis_latest_year

    def generate_plots(self) -> None:
        """Renders stacked bar/line plots showcasing multi-year DuPont factor breakdown."""
        print("Generating Dupoint Graphs......")
        self.graph = graph_dupoint_analysispy.Graph(
            fetcher=self.return_on_capital)
        self.graph.plot()


class Zscore():
    """Orchestration engine evaluating solvency and insolvency risk using Altman Z-Score models.

    Consumes balance sheet and income statement metrics (Working Capital, Retained Earnings, 
    EBIT, Market Cap, Total Liabilities/Assets) to assess 2-year distress probabilities.

    Attributes:
        calculation (Calculation_Altman_zscore.Calculation): Weighting engine for Z-score formulas.
        financial_health (Optional[FinancialHealth]): Assessor categorizing Safe, Grey, or Distress zones.
        final_value (Optional[Any]): Summary output container.
        graph (Optional[graph_zscore.Graph]): Matplotlib plotter showing historical Z-score trends.
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Injects clean financial data into the Altman Z-score calculation module.

        Args:
            fetcher (StoringAndCleaning): Injected data container providing cleaned DataFrames.
        """
        self.calculation: Calculation_Altman_zscore.Calculation = Calculation_Altman_zscore.Calculation(
            fetcher=fetcher)
        self.financial_health: Optional[FinancialHealth] = None
        self.final_value: Optional[Any] = None
        self.graph: Optional[graph_zscore.Graph] = None

    def master_initialization(self) -> None:
        """Executes the full Altman Z-Score pipeline from math calculation to plot generation."""
        self.calculation_initializer()
        self.calc_financial_health()
        self.generate_plot()
        print("Zscore Pipeline execution complete.")

    def calculation_initializer(self) -> None:
        """Triggers mathematical formula computation for individual Z-score sub-ratios."""
        print("Running Zscore Calculations.......")
        self.calculation.initializer()

    def calc_financial_health(self) -> None:
        """Evaluates overall corporate solvency health and zone categorization."""
        print("Calculating Financial Health")
        self.financial_health = FinancialHealth(fetcher=self.calculation)
        self.financial_health.fetch_all_data()

    def generate_plot(self) -> None:
        """Plots historical Altman Z-Score lines alongside safety thresholds (1.81 and 2.99)."""
        self.graph = graph_zscore.Graph(fetcher=self.financial_health)
        self.graph.plot()
        self.final_value = {
            "score": self.financial_health.score,
            "zone": self.financial_health.zone_label(),
        }


class CCC():
    """Orchestration engine analyzing working capital efficiency via the Cash Conversion Cycle (CCC).

    Measures time (in days) required to convert inventory investments and operational inputs 
    back into cash inflows. Synthesizes Days Inventory Outstanding (DIO), Days Sales 
    Outstanding (DSO), and Days Payable Outstanding (DPO).

    Attributes:
        calculation (Calculations_CCC.Calculation): Math engine deriving DIO, DSO, and DPO metrics.
        operational_efficiency (Optional[OperationEFFICIENCY]): Module synthesizing working capital efficiency.
        graph (Optional[CCC_graph.Graph]): Chart engine visualizing working capital trends over time.
        final_value (Optional[Any]): Final CCC value storage.
    """

    def __init__(self, fetcher: StoringAndCleaning) -> None:
        """Injects clean DataFrame sources into the CCC operational efficiency pipeline.

        Args:
            fetcher (StoringAndCleaning): Cleaned financial DataFrame instance.
        """
        self.calculation: Calculations_CCC.Calculation = Calculations_CCC.Calculation(
            fetcher=fetcher)
        self.operational_efficiency: Optional[OperationEFFICIENCY] = None
        self.graph: Optional[CCC_graph.Graph] = None
        self.final_value: Optional[Any] = None

    def master_initialization(self) -> None:
        """Runs the CCC execution pipeline in strict operational order."""
        self.calculation_initializer()
        self.calc_operational_efficency()
        self.graph_plot()
        print("Cash Conversion Cycle Pipeline execution complete.")

    def calculation_initializer(self) -> None:
        """Calculates operational cycle day metrics (DIO, DSO, DPO)."""
        print("Running CCC Calculations.......")
        self.calculation.master_initializer()

    def calc_operational_efficency(self) -> None:
        """Combines component metrics (`DIO + DSO - DPO`) to arrive at net Cash Conversion Cycle days."""
        print("Calculating Operational Efficiency")
        self.operational_efficiency = OperationEFFICIENCY(
            fetcher=self.calculation)
        self.operational_efficiency.calc()
        self.final_value = self.operational_efficiency.ccc_latest_year

    def graph_plot(self) -> None:
        """Renders graphical visualizations of working capital cycles."""
        print("Plotting Cash Conversion Cycle....")
        self.graph = CCC_graph.Graph(fetcher=self.operational_efficiency)
        self.graph.plot()
