"""Browser UI for equity analysis. All inputs are widgets — no terminal prompts."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
_APP = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

try:
    if "API" in st.secrets:
        os.environ["API"] = str(st.secrets["API"])
except Exception:
    pass

from tickers import DEFAULT_TICKER, TICKERS  # noqa: E402
from analysis_model.orchestrators import tracker_status  # noqa: E402
from analysis_model.pipeline import run_pipeline  # noqa: E402

SESSION_RUN_CAP = 15

st.set_page_config(
    page_title="Equity Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)


def _require_password() -> None:
    expected = _secret("APP_PASSWORD")
    if not expected:
        return
    if st.session_state.get("authed"):
        return
    st.title("Equity analysis")
    entered = st.text_input("Shared password", type="password")
    if st.button("Unlock"):
        if entered == expected:
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def _selectbox_kwargs() -> Dict[str, Any]:
    params = inspect.signature(st.selectbox).parameters
    kwargs: Dict[str, Any] = {}
    if "accept_new_options" in params:
        kwargs["accept_new_options"] = True
    return kwargs


_require_password()

if "session_runs" not in st.session_state:
    st.session_state.session_runs = 0

quota = tracker_status()

st.title("Equity analysis")
st.caption(
    "Altman Z-Score, DuPont ROE, cash conversion cycle, and DCF"
)

with st.sidebar:
    default_index = TICKERS.index(DEFAULT_TICKER) if DEFAULT_TICKER in TICKERS else 0
    ticker = st.selectbox(
        "Ticker",
        options=TICKERS,
        index=default_index,
        help="Type to filter. Names not on the list can be typed if this Streamlit version allows it.",
        **_selectbox_kwargs(),
    )
    st.subheader("Analyses")
    run_zscore = st.checkbox("Z-Score", value=True)
    run_dupont = st.checkbox("DuPont", value=True)
    run_ccc = st.checkbox("CCC", value=True)
    run_dcf = st.checkbox("DCF", value=True)
    st.subheader("DCF assumptions")
    dcf_growth_pct = st.number_input(
        "Perpetual growth (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Terminal growth rate. Default 2% matches the original model.",
    )
    dcf_years = st.number_input(
        "Projection years",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Discrete forecast horizon. Default 5 years matches the original model.",
    )
    st.caption(
        f"Daily uncached runs left: {quota['remaining']} / {quota['max_calls']}. "
        "One run is one analysis (about six FMP requests). Cached tickers today do not count."
    )
    st.caption(f"This session: {st.session_state.session_runs} / {SESSION_RUN_CAP} runs.")
    submitted = st.button("Run analysis", type="primary", use_container_width=True)


def _metric_items(metrics: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    if metrics.get("zscore") is not None:
        zone = metrics.get("zscore_zone") or ""
        label = "Z-Score" + (f" ({zone})" if zone else "")
        items.append(
            (
                label,
                f"{metrics['zscore']:.2f}",
                "Altman Z-Score. Distress / grey / safe uses the model zone cutoffs.",
            )
        )
    if metrics.get("roe") is not None:
        items.append(
            (
                "DuPont ROE",
                f"{metrics['roe']:.2%}",
                "Latest-year five-step DuPont ROE (product of the five drivers).",
            )
        )
    if metrics.get("ccc") is not None:
        items.append(
            (
                "CCC (days)",
                f"{metrics['ccc']:.1f}",
                "Latest-year cash conversion cycle: DIO + DSO − DPO.",
            )
        )
    if metrics.get("valuation"):
        short = metrics.get("valuation_short") or "See details"
        items.append(
            (
                "DCF",
                str(short),
                str(metrics["valuation"]),
            )
        )
    return items


if submitted:
    if not (run_zscore or run_dupont or run_ccc or run_dcf):
        st.warning("Select at least one analysis.")
    elif not os.getenv("API"):
        st.error(
            "No API key found. Locally, copy `.env.example` to `api.env` and set `API=`. "
            "On Streamlit Community Cloud, add a secret named `API`."
        )
    elif st.session_state.session_runs >= SESSION_RUN_CAP:
        st.error("This browser session hit the run cap. Refresh the page to start a new session.")
    else:
        symbol = str(ticker).strip().upper()
        with st.spinner(f"Running analysis for {symbol}…"):
            result: Optional[Dict[str, Any]] = run_pipeline(
                ticker=symbol,
                run_zscore=run_zscore,
                run_dupont=run_dupont,
                run_ccc=run_ccc,
                run_dcf=run_dcf,
                dcf_growth=float(dcf_growth_pct) / 100.0,
                dcf_years=int(dcf_years),
            )

        if result is not None:
            err: Optional[str] = result.get("error")
            if err:
                st.error(err)
            else:
                st.session_state.session_runs += 1
                if result.get("from_cache"):
                    st.caption("Statements loaded from today's cache (no extra FMP calls).")
                for warning in result.get("warnings") or []:
                    st.warning(warning)

                metrics = result.get("metrics") or {}
                figures = result.get("figures") or {}
                items = _metric_items(metrics)
                if items:
                    cols = st.columns(len(items))
                    for col, (label, value, help_text) in zip(cols, items):
                        col.metric(label, value, help=help_text)
                    if metrics.get("valuation"):
                        st.caption(metrics["valuation"])

                if figures.get("zscore") is not None:
                    st.subheader("Z-Score")
                    st.pyplot(figures["zscore"], use_container_width=True)
                if figures.get("dupont") is not None:
                    st.subheader("DuPont")
                    st.pyplot(figures["dupont"], use_container_width=True)
                if figures.get("ccc") is not None:
                    st.subheader("Cash conversion cycle")
                    st.pyplot(figures["ccc"], use_container_width=True)
                if figures.get("dcf") is not None:
                    st.subheader("Discounted cash flow")
                    st.pyplot(figures["dcf"], use_container_width=True)
else:
    st.info("Choose a ticker and analyses in the sidebar, then click **Run analysis**.")
