"""Browser UI for equity analysis. All inputs are widgets — no terminal prompts."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    if "API" in st.secrets:
        os.environ["API"] = str(st.secrets["API"])
except Exception:
    pass

from analysis_model.pipeline import run_pipeline  # noqa: E402

st.set_page_config(
    page_title="Equity Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Equity analysis")
st.caption("Altman Z-Score, DuPont ROE, cash conversion cycle, and DCF — same models as the original scripts.")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL", help="Exchange ticker, e.g. AAPL")
    st.subheader("Analyses")
    run_zscore = st.checkbox("Z-Score", value=True)
    run_dupont = st.checkbox("DuPont", value=True)
    run_ccc = st.checkbox("CCC", value=True)
    run_dcf = st.checkbox("DCF", value=True)
    submitted = st.button("Run analysis", type="primary", use_container_width=True)

if submitted:
    if not (run_zscore or run_dupont or run_ccc or run_dcf):
        st.warning("Select at least one analysis.")
    elif not os.getenv("API"):
        st.error(
            "No API key found. Locally, copy `.env.example` to `api.env` and set `API=`. "
            "On Streamlit Community Cloud, add a secret named `API`."
        )
    else:
        with st.spinner(f"Running analysis for {ticker.strip().upper()}…"):
            try:
                result: Dict[str, Any] = run_pipeline(
                    ticker=ticker,
                    run_zscore=run_zscore,
                    run_dupont=run_dupont,
                    run_ccc=run_ccc,
                    run_dcf=run_dcf,
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                result = None

        if result is not None:
            err: Optional[str] = result.get("error")
            if err:
                st.error(err)
            else:
                metrics = result.get("metrics") or {}
                figures = result.get("figures") or {}
                cols = st.columns(4)
                if "zscore" in metrics:
                    cols[0].metric("Altman Z-Score", f"{metrics['zscore']:.2f}")
                if "roe" in metrics and metrics["roe"] is not None:
                    cols[1].metric("DuPont ROE (latest)", f"{metrics['roe']:.2%}")
                if "ccc" in metrics and metrics["ccc"] is not None:
                    cols[2].metric("CCC (latest, days)", f"{metrics['ccc']:.1f}")
                if "valuation" in metrics and metrics["valuation"]:
                    cols[3].metric("DCF verdict", str(metrics["valuation"]))

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
