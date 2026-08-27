"""Thin wrapper around existing orchestrators for the Streamlit UI and CLI."""

from typing import Any, Dict

from analysis_model.orchestrators import (
    CCC,
    DiscountedCashFlow,
    Dupoint,
    GatheringAndStoringAndCleaning,
    Zscore,
)


def run_pipeline(
    ticker: str,
    run_zscore: bool = True,
    run_dupont: bool = True,
    run_ccc: bool = True,
    run_dcf: bool = True,
) -> Dict[str, Any]:
    """Fetch data once, then run the selected analysis modules.

    Returns a dict with ticker, optional error, metrics, and matplotlib figures.
    """
    symbol: str = ticker.strip().upper()
    result: Dict[str, Any] = {
        "ticker": symbol,
        "error": None,
        "metrics": {},
        "figures": {},
    }

    if not symbol:
        result["error"] = "Enter a ticker symbol."
        return result

    data = GatheringAndStoringAndCleaning(ticker=symbol)
    data.initialize_data()
    if not data.api_limit:
        result["error"] = "Daily API call limit reached. Try again tomorrow."
        return result

    data.process_data()

    if run_zscore:
        zscore = Zscore(fetcher=data.storing)
        zscore.master_initialization()
        result["metrics"]["zscore"] = zscore.financial_health.score
        result["figures"]["zscore"] = zscore.graph.fig

    if run_dupont:
        dupont = Dupoint(fetcher=data.storing)
        dupont.master_initializer()
        result["metrics"]["roe"] = dupont.return_on_capital.dupoint_analysis_latest_year
        result["figures"]["dupont"] = dupont.graph.fig

    if run_ccc:
        ccc = CCC(fetcher=data.storing)
        ccc.master_initialization()
        result["metrics"]["ccc"] = ccc.operational_efficiency.ccc_latest_year
        result["figures"]["ccc"] = ccc.graph.fig

    if run_dcf:
        dcf = DiscountedCashFlow(data=data.storing)
        dcf.master_initialization()
        result["metrics"]["valuation"] = dcf.main_value
        result["figures"]["dcf"] = dcf.graph.fig

    return result
