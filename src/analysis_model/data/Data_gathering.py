"""Network Ingestion Layer for Equity and Macroeconomic Data.

This module communicates with external APIs (Financial Modeling Prep and Yahoo Finance via `yfinance`)
to retrieve historical financial statements, company profile metrics, market risk premiums, and US Treasury yields.

Data Flow:
    Ticker Symbol -> `DataGathering` -> HTTP Requests -> JSON Payloads -> `StoringAndCleaning`
"""

from datetime import date, timedelta
import os
from typing import Any, Dict, List, Optional, Union

import requests as r
from dotenv import load_dotenv
import yfinance as yf

from analysis_model.errors import AnalysisError

# Load API from project-root api.env if present (Streamlit Cloud uses env / secrets instead)
_search_dir: str = os.path.dirname(os.path.abspath(__file__))
for _ in range(6):
    _candidate: str = os.path.join(_search_dir, "api.env")
    if os.path.isfile(_candidate):
        load_dotenv(_candidate)
        break
    _parent: str = os.path.dirname(_search_dir)
    if _parent == _search_dir:
        break
    _search_dir = _parent

# Type alias representing raw API JSON responses (lists or dictionaries)
JSONResponse = Union[List[Dict[str, Any]], Dict[str, Any]]


class DataGathering:
    """Handles external API interactions to fetch raw financial statements and market metrics.

    Communicates directly with Financial Modeling Prep (FMP) and Yahoo Finance to assemble
    the fundamental inputs required for valuation and risk modeling.

    Attributes:
        api_key (Optional[str]): The authentication token retrieved from environment variables.
        stock_symbol (str): Target company ticker symbol (e.g., "AAPL").
        parameters (Dict[str, Union[str, int, Optional[str]]]): Shared parameter dictionary for FMP API calls.
        pnl_data (Optional[JSONResponse]): Income Statement response payload.
        balance_sheet_data (Optional[JSONResponse]): Balance Sheet response payload.
        cashflow_data (Optional[JSONResponse]): Cash Flow Statement response payload.
        company_profile_data (Optional[JSONResponse]): Profile payload (Beta, Market Cap).
        treasury_rate_data (Optional[JSONResponse]): Recent US Treasury yield curve data.
        market_premium_data (Optional[JSONResponse]): Country-specific market risk premium data.
        general_data (Optional[Dict[str, Any]]): Raw dictionary from Yahoo Finance (`yfinance.Ticker.info`).
        country (Optional[str]): Company country of incorporation.
        sector (Optional[str]): Industry sector classification.
    """

    def __init__(self, symbol: str) -> None:
        """Initializes the network client for a given equity ticker.

        Args:
            symbol (str): The financial stock ticker symbol.
        """
        self.api_key: Optional[str] = os.getenv("API")
        self.stock_symbol: str = symbol

        # Shared query parameters used across financial statement endpoints
        self.parameters: Dict[str, Union[str, int, Optional[str]]] = {
            "symbol": self.stock_symbol,
            "limit": 5,
            "period": "annual",
            "apikey": self.api_key,
        }

        # Raw state containers initialized to None until explicitly populated by fetch_all_data()
        self.pnl_data: Optional[JSONResponse] = None
        self.balance_sheet_data: Optional[JSONResponse] = None
        self.cashflow_data: Optional[JSONResponse] = None
        self.company_profile_data: Optional[JSONResponse] = None
        self.treasury_rate_data: Optional[JSONResponse] = None
        self.market_premium_data: Optional[JSONResponse] = None
        self.general_data: Optional[Dict[str, Any]] = None

        self.country: Optional[str] = None
        self.sector: Optional[str] = None

    def _fmp_get(self, url: str, params: Dict[str, Any]) -> JSONResponse:
        """GET an FMP endpoint and map HTTP/empty responses to AnalysisError."""
        if not self.api_key:
            raise AnalysisError(
                "No API key found. Set API in api.env (local) or Streamlit secrets."
            )
        try:
            response: r.Response = r.get(url=url, params=params, timeout=30)
        except r.RequestException as exc:
            raise AnalysisError(
                "Could not reach Financial Modeling Prep. Check your network and try again."
            ) from exc

        if response.status_code in (401, 403):
            raise AnalysisError(
                "API key was rejected. Check the FMP key in secrets or api.env."
            )
        if response.status_code == 429:
            raise AnalysisError(
                "Financial Modeling Prep rate limit reached. Try again later."
            )
        if response.status_code >= 400:
            raise AnalysisError(
                f"No usable data for {self.stock_symbol} (HTTP {response.status_code}). "
                "The ticker may be invalid, delisted, or unsupported."
            )

        try:
            payload: JSONResponse = response.json()
        except ValueError as exc:
            raise AnalysisError(
                f"Unexpected response for {self.stock_symbol}. Try another ticker."
            ) from exc

        if isinstance(payload, dict) and payload.get("Error Message"):
            raise AnalysisError(
                f"No data for {self.stock_symbol}: {payload.get('Error Message')}"
            )
        if payload in (None, [], {}):
            raise AnalysisError(
                f"No financial statements for {self.stock_symbol}. "
                "Check the ticker (ETFs and delisted names often fail)."
            )
        return payload

    def to_payload(self) -> Dict[str, Any]:
        """Serialize fetched payloads for disk / Streamlit cache."""
        return {
            "symbol": self.stock_symbol,
            "pnl_data": self.pnl_data,
            "balance_sheet_data": self.balance_sheet_data,
            "cashflow_data": self.cashflow_data,
            "company_profile_data": self.company_profile_data,
            "treasury_rate_data": self.treasury_rate_data,
            "market_premium_data": self.market_premium_data,
            "general_data": self.general_data,
            "country": self.country,
            "sector": self.sector,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DataGathering":
        """Rebuild a gathering instance from a cached payload (no HTTP)."""
        instance = cls(symbol=str(payload.get("symbol", "")))
        instance.pnl_data = payload.get("pnl_data")
        instance.balance_sheet_data = payload.get("balance_sheet_data")
        instance.cashflow_data = payload.get("cashflow_data")
        instance.company_profile_data = payload.get("company_profile_data")
        instance.treasury_rate_data = payload.get("treasury_rate_data")
        instance.market_premium_data = payload.get("market_premium_data")
        instance.general_data = payload.get("general_data")
        instance.country = payload.get("country")
        instance.sector = payload.get("sector")
        return instance

    def fetch_all_data(self) -> None:
        """Executes sequential API calls to populate all financial and market properties."""
        self.pnl_data = self.income_statement()
        self.balance_sheet_data = self.balance_sheet()
        self.cashflow_data = self.cashflow()
        self.company_profile_data = self.company_profile()
        self.treasury_rate_data = self.treasury_rate()
        self.market_premium_data = self.market_premium()
        self.general_info()

    def income_statement(self) -> JSONResponse:
        """Fetches the 5-year historical Income Statement (P&L) data from FMP.

        Returns:
            JSONResponse: List of dictionaries containing Revenue, EBIT, Taxes, and Net Income.
        """
        pnl_url: str = "https://financialmodelingprep.com/stable/income-statement"
        return self._fmp_get(url=pnl_url, params=dict(self.parameters))

    def balance_sheet(self) -> JSONResponse:
        """Fetches the 5-year historical Balance Sheet data from FMP.

        Returns:
            JSONResponse: List of dictionaries containing Total Assets, Liabilities, and Equity items.
        """
        bs_url: str = "https://financialmodelingprep.com/stable/balance-sheet-statement"
        return self._fmp_get(url=bs_url, params=dict(self.parameters))

    def cashflow(self) -> JSONResponse:
        """Fetches the 5-year historical Cash Flow Statement from FMP.

        Returns:
            JSONResponse: List of dictionaries containing Operating Cash Flows and CapEx.
        """
        cf_url: str = "https://financialmodelingprep.com/stable/cash-flow-statement"
        return self._fmp_get(url=cf_url, params=dict(self.parameters))

    def company_profile(self) -> JSONResponse:
        """Fetches key profile metrics (Beta, Market Capitalization) from FMP.

        Returns:
            JSONResponse: Dictionary containing overview market parameters.
        """
        profile_url: str = "https://financialmodelingprep.com/stable/profile"
        return self._fmp_get(url=profile_url, params=dict(self.parameters))

    def treasury_rate(self) -> JSONResponse:
        """Fetches US Treasury bond yields using a 5-day historical lookback window.

        Design Rationale:
            Bond markets are closed on weekends and holidays. A 5-day window guarantees
            a valid yield response regardless of the calendar day the code is executed.

        Returns:
            JSONResponse: Treasury yield rates over the lookback window.
        """
        rate_url: str = "https://financialmodelingprep.com/stable/treasury-rates"
        start_date: date = date.today() - timedelta(days=5)

        # Overrides global parameters to query macro economic yields
        parameters: Dict[str, Optional[str]] = {
            "apikey": self.api_key,
            "from": start_date.isoformat(),
            "to": date.today().isoformat(),
        }
        return self._fmp_get(url=rate_url, params=parameters)

    def market_premium(self) -> JSONResponse:
        """Fetches regional Equity Market Risk Premiums (ERP) for CAPM hurdle calculations.

        Returns:
            JSONResponse: Country and regional equity risk premium rates.
        """
        premium_url: str = "https://financialmodelingprep.com/stable/market-risk-premium"
        parameters: Dict[str, Optional[str]] = {"apikey": self.api_key}
        return self._fmp_get(url=premium_url, params=parameters)

    def general_info(self) -> None:
        """Extracts general metadata (country, industry sector) using `yfinance`.

        Design Rationale:
            `yfinance` provides reliable sector and country metadata, supplementing FMP financial statements.
        """
        try:
            ticker: yf.Ticker = yf.Ticker(self.stock_symbol)
            information: Dict[str, Any] = ticker.info or {}
        except Exception as exc:
            raise AnalysisError(
                f"Could not load company profile for {self.stock_symbol}."
            ) from exc

        self.general_data = information
        self.country = information.get("country")
        self.sector = information.get("sector")

        if isinstance(self.company_profile_data, list) and self.company_profile_data:
            profile_row = self.company_profile_data[0]
            if not self.country:
                self.country = profile_row.get("country")
            if not self.sector:
                self.sector = profile_row.get("sector")
        elif isinstance(self.company_profile_data, dict):
            if not self.country:
                self.country = self.company_profile_data.get("country")
            if not self.sector:
                self.sector = self.company_profile_data.get("sector")


if __name__ == "__main__":
    # Integration smoke test: Ingest AAPL network responses and print summaries
    data: DataGathering = DataGathering("AAPL")
    data.fetch_all_data()

    print("----------------PNL-------------------------")
    print(data.pnl_data)
    print("----------------BS-------------------------")
    print(data.balance_sheet_data)
    print("----------------CF-------------------------")
    print(data.cashflow_data)
    print("----------------CPD-------------------------")
    print(data.company_profile_data)
    print("----------------Rate-------------------------")
    print(data.treasury_rate_data)
    print("----------------Premium----------------------")
    print(data.market_premium_data)
