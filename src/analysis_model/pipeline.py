"""Thin wrapper around existing orchestrators for the Streamlit UI and CLI."""

from typing import Any, Dict, Optional

from analysis_model.errors import AnalysisError
from analysis_model.orchestrators import (
    CCC,
    DiscountedCashFlow,
    Dupoint,
    GatheringAndStoringAndCleaning,
    Zscore,
    tracker_status,
)


def short_dcf_verdict(detail: Optional[str]) -> str:
    if not detail:
        return "—"
    lower = detail.lower()
    if "undervalued" in lower:
        return "Undervalued"
    if "overvalued" in lower:
        return "Overvalued"
    if "fair" in lower:
        return "Fair"
    return "See details"


def run_pipeline(
    ticker: str,
    run_zscore: bool = True,
    run_dupoint: bool = True,
    run_ccc: bool = True,
    run_dcf: bool = True,
    dcf_growth: float = 0.02,
) -> Dict[str, Any]:
    """Fetch data once, then run the selected analysis modules.

    Returns a dict with ticker, optional error, metrics, figures, and warnings.
    """
    symbol: str = ticker.strip().upper()
    result: Dict[str, Any] = {
        "ticker": symbol,
        "error": None,
        "metrics": {},
        "figures": {},
        "warnings": [],
        "from_cache": False,
        "tracker": tracker_status(),
    }

    if not symbol:
        result["error"] = "Enter a ticker symbol."
        return result

    try:
        data = GatheringAndStoringAndCleaning(ticker=symbol)
        data.initialize_data()
        if not data.api_limit:
            result["error"] = (
                "Daily analysis-run limit reached (30 uncached runs per day; "
                "each run uses about six FMP requests). Try again tomorrow, "
                "or re-run a ticker already cached today."
            )
            result["tracker"] = tracker_status()
            return result

        data.process_data()
        result["from_cache"] = data.from_cache
        if data.premium_warning:
            result["warnings"].append(data.premium_warning)

        if run_zscore:
            zscore = Zscore(fetcher=data.storing)
            zscore.master_initialization()
            result["metrics"]["zscore"] = zscore.financial_health.score
            result["metrics"]["zscore_zone"] = zscore.financial_health.zone_label()
            result["metrics"]["zscore_final"] = zscore.final_value
            result["figures"]["zscore"] = zscore.graph.fig

        if run_dupoint:
            dupoint = Dupoint(fetcher=data.storing)
            dupoint.master_initializer()
            result["metrics"]["roe"] = dupoint.final_value
            result["figures"]["dupoint"] = dupoint.graph.fig

        if run_ccc:
            ccc = CCC(fetcher=data.storing)
            ccc.master_initialization()
            result["metrics"]["ccc"] = ccc.final_value
            result["figures"]["ccc"] = ccc.graph.fig

        if run_dcf:
            dcf = DiscountedCashFlow(
                data=data.storing,
                const_growth_rate=dcf_growth,
            )
            dcf.master_initialization()
            result["metrics"]["valuation"] = dcf.main_value
            result["metrics"]["valuation_short"] = short_dcf_verdict(dcf.main_value)
            result["figures"]["dcf"] = dcf.graph.fig

        result["tracker"] = tracker_status()
        return result
    except AnalysisError as exc:
        result["error"] = str(exc)
        result["tracker"] = tracker_status()
        return result
