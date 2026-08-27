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
        pnl: r.Response = r.get(url=pnl_url, params=self.parameters)
        pnl.raise_for_status()
        return pnl.json()

    def balance_sheet(self) -> JSONResponse:
        """Fetches the 5-year historical Balance Sheet data from FMP.

        Returns:
            JSONResponse: List of dictionaries containing Total Assets, Liabilities, and Equity items.
        """
        bs_url: str = "https://financialmodelingprep.com/stable/balance-sheet-statement"
        bs: r.Response = r.get(url=bs_url, params=self.parameters)
        bs.raise_for_status()
        return bs.json()

    def cashflow(self) -> JSONResponse:
        """Fetches the 5-year historical Cash Flow Statement from FMP.

        Returns:
            JSONResponse: List of dictionaries containing Operating Cash Flows and CapEx.
        """
        cf_url: str = "https://financialmodelingprep.com/stable/cash-flow-statement"
        cf: r.Response = r.get(url=cf_url, params=self.parameters)
        cf.raise_for_status()
        return cf.json()

    def company_profile(self) -> JSONResponse:
        """Fetches key profile metrics (Beta, Market Capitalization) from FMP.

        Returns:
            JSONResponse: Dictionary containing overview market parameters.
        """
        profile_url: str = "https://financialmodelingprep.com/stable/profile"
        profile: r.Response = r.get(url=profile_url, params=self.parameters)
        profile.raise_for_status()
        return profile.json()

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
        rate: r.Response = r.get(url=rate_url, params=parameters)
        rate.raise_for_status()
        return rate.json()

    def market_premium(self) -> JSONResponse:
        """Fetches regional Equity Market Risk Premiums (ERP) for CAPM hurdle calculations.

        Returns:
            JSONResponse: Country and regional equity risk premium rates.
        """
        premium_url: str = "https://financialmodelingprep.com/stable/market-risk-premium"
        parameters: Dict[str, Optional[str]] = {"apikey": self.api_key}
        premium: r.Response = r.get(url=premium_url, params=parameters)
        premium.raise_for_status()
        return premium.json()

    def general_info(self) -> None:
        """Extracts general metadata (country, industry sector) using `yfinance`.

        Design Rationale:
            `yfinance` provides reliable sector and country metadata, supplementing FMP financial statements.
        """
        ticker: yf.Ticker = yf.Ticker(self.stock_symbol)
        information: Dict[str, Any] = ticker.info

        self.general_data = information
        self.country = information.get("country")
        self.sector = information.get("sector")


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
