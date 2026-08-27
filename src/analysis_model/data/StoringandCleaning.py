import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TypedDict, Tuple
import pandas as pd
import numpy as np

from analysis_model.errors import AnalysisError


class ProfileDict(TypedDict):
    """Structural type representation for company profile items."""
    price: float
    beta: float
    marketCap: float


class TreasuryDict(TypedDict):
    """Structural type representation for treasury rate records."""
    year10: float


class RiskPremiumDict(TypedDict):
    """Structural type representation for equity risk premium metrics."""
    totalEquityRiskPremium: float


# Structural JSON Payload Aliases
FinancialRecord = Dict[str, Any]
JSONData = Union[List[FinancialRecord], FinancialRecord]


_COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "south korea": "korea",
    "republic of korea": "korea",
}


def normalize_country(name: Optional[str]) -> str:
    raw = (name or "").strip().lower()
    return _COUNTRY_ALIASES.get(raw, raw)


def match_equity_risk_premium(
    premium_rows: Any, country: Optional[str]
) -> Tuple[float, Optional[str]]:
    """Return (ERP, warning). Falls back to United States if country is missing or unmatched."""
    rows: List[Dict[str, Any]]
    if isinstance(premium_rows, list):
        rows = [row for row in premium_rows if isinstance(row, dict)]
    elif isinstance(premium_rows, dict):
        rows = [premium_rows]
    else:
        rows = []

    def _premium_of(row: Dict[str, Any]) -> Optional[float]:
        value = row.get("totalEquityRiskPremium")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _country_of(row: Dict[str, Any]) -> str:
        return normalize_country(str(row.get("country") or row.get("Country") or ""))

    us_rate: Optional[float] = None
    for row in rows:
        if _country_of(row) == "united states":
            us_rate = _premium_of(row)
            break

    target = normalize_country(country)
    if target:
        for row in rows:
            if _country_of(row) == target:
                rate = _premium_of(row)
                if rate is not None:
                    return rate, None

    if us_rate is not None:
        warning = (
            f"No equity risk premium match for country {country!r}; using United States."
            if target
            else "Company country missing; using United States equity risk premium."
        )
        return us_rate, warning

    if rows:
        fallback = _premium_of(rows[0])
        if fallback is not None:
            return fallback, "Could not match country ERP; using first premium row."

    raise AnalysisError("Equity risk premium data is empty.")


class StoringAndCleaning:
    """Extracts, filters, and formats raw JSON financial payloads into clean Pandas DataFrames.

    Architectural Role:
    -------------------
    This class serves as the Transformation and Loading (TL) layer of the financial ETL pipeline. 
    It bridges the gap between unstructured API data payloads (provided by `DataGathering`) and 
    downstream valuation/financial calculation models. 

    Data Flow:
    ----------
    1. Input: Receives raw JSON-formatted lists/dictionaries from an active `DataGathering` instance.
    2. Extraction: Iterates through row mappings using `data_extractor()` to generate chronological 1D vectors.
    3. Alignment: Maps parsed financial metrics against standardized fiscal years (`self.fy`).
    4. Output: Bundles vectors into Pandas DataFrames indexed by fiscal year (`df_pnl`, `df_bs`, `df_cf`) 
       and a single-row snapshot summary (`df_extra`).
    """

    def __init__(self, data_fetch: Any) -> None:
        """Initializes data structures and loads raw data payloads from the fetcher.

        Args:
            data_fetch (DataGathering): An active instance of `DataGathering` containing 
                populated financial API responses.
        """
        # --- Raw JSON API Responses ---
        self.pnl: List[FinancialRecord] = data_fetch.pnl_data
        self.bs: List[FinancialRecord] = data_fetch.balance_sheet_data
        self.cf: List[FinancialRecord] = data_fetch.cashflow_data
        self.profile: List[ProfileDict] = data_fetch.company_profile_data
        self.rate: List[TreasuryDict] = data_fetch.treasury_rate_data
        self.premium: List[RiskPremiumDict] = data_fetch.market_premium_data

        # --- Extracted Time Indices ---
        self.fy: List[int] = []

        # --- Parsed Income Statement Vectors ---
        self.revenue: List[float] = []
        self.net_income: List[float] = []           # Stores EBIT
        self.interest_expense: List[float] = []
        self.tax: List[float] = []
        self.operation_income: List[float] = []
        self.income_before_tax: List[float] = []
        self.depreciation: List[float] = []
        self.net_income_: List[float] = []          # Stores actual Net Income
        self.cogs: List[float] = []

        # --- Parsed Balance Sheet Vectors ---
        self.total_debt: List[float] = []
        self.cash_and_cash_equivalents: List[float] = []
        self.inventory: List[float] = []
        self.accounts_receviables: List[float] = []
        self.accounts_payable: List[float] = []
        self.total_equity: List[float] = []
        self.total_assets: List[float] = []
        self.retained_earning: List[float] = []
        self.total_liabilities: List[float] = []
        self.total_current_liabilities: np.ndarray = np.array([])
        self.total_current_assets: np.ndarray = np.array([])

        # --- Parsed Cash Flow Vectors ---
        self.cash_from_operation: List[float] = []
        self.capex: List[float] = []
        self.credit_sales: List[float] = []

        # --- Valuation Snapshot Variables ---
        self.current_price: float = 0.0
        self.beta: float = 0.0
        self.market_cap: float = 0.0
        self.interest_rate: float = 0.0
        self.dilluted_shares: float = 0.0
        self.premium_rate: float = 0.0
        self.premium_warning: Optional[str] = None

        # --- Structured Output DataFrames ---
        self.df_pnl: pd.DataFrame = pd.DataFrame()
        self.df_bs: pd.DataFrame = pd.DataFrame()
        self.df_cf: pd.DataFrame = pd.DataFrame()
        self.df_extra: pd.DataFrame = pd.DataFrame()

        # --- Metadata Attributes ---
        self.country: str = getattr(data_fetch, "country", "")
        self.sector: str = getattr(data_fetch, "sector", "")

    def fetch_dataframe(self) -> None:
        """Orchestrates the entire extraction pipeline to construct clean DataFrames.

        Executes the extraction pipeline sequentially:
        1. Parse reporting years (`financial_years`).
        2. Extract line items for all three statements.
        3. Pull market constants (`extras`).
        4. Construct final Pandas DataFrames.
        """
        if not self.pnl or not self.bs or not self.cf:
            raise AnalysisError(
                "Income statement, balance sheet, or cash flow is empty. "
                "The ticker may be invalid, an ETF, or delisted."
            )
        self.financial_years()
        self.income_statement()
        self.balance_sheet()
        self.cashflow()
        self.extras()
        self.pandas_dataframe()

    def data_extractor(self, variable: List[Dict[str, Any]], date_required: str) -> List[Any]:
        """Extracts a target key across a list of financial JSON mapping objects.

        Why this exists:
        ----------------
        API data arrives as a list of dictionary records per year. This helper collapses 
        a single key across all years into a clean 1D primitive list.

        Args:
            variable (List[Dict[str, Any]]): List of financial statement records.
            date_required (str): Key name to target in each dictionary.

        Returns:
            List[Any]: Extracted list containing chronological metric values.
        """
        return [item[date_required] for item in variable]

    def financial_years(self) -> List[int]:
        """Extracts reporting years and converts them to integers for DataFrame indexing.

        Returns:
            List[int]: Parsed list of fiscal years (e.g., [2023, 2022, 2021]).
        """
        raw_years: List[Any] = self.data_extractor(
            variable=self.pnl, date_required="fiscalYear"
        )
        self.fy = [int(item) for item in raw_years]
        return self.fy

    def income_statement(self) -> None:
        """Parses historical Income Statement metrics into internal 1D lists."""
        self.revenue = self.data_extractor(
            variable=self.pnl, date_required="revenue")
        self.net_income = self.data_extractor(
            variable=self.pnl, date_required="ebit")
        self.interest_expense = self.data_extractor(
            variable=self.pnl, date_required="interestExpense")
        self.tax = self.data_extractor(
            variable=self.pnl, date_required="incomeTaxExpense")
        self.operation_income = self.data_extractor(
            variable=self.pnl, date_required="operatingIncome")
        self.income_before_tax = self.data_extractor(
            variable=self.pnl, date_required="incomeBeforeTax")
        self.depreciation = self.data_extractor(
            variable=self.pnl, date_required="depreciationAndAmortization")
        self.net_income_ = self.data_extractor(
            variable=self.pnl, date_required="netIncome")
        self.cogs = self.data_extractor(
            variable=self.pnl, date_required="costOfRevenue")

    def balance_sheet(self) -> None:
        """Parses historical Balance Sheet metrics into internal 1D lists and NumPy arrays."""
        self.total_debt = self.data_extractor(
            variable=self.bs, date_required="totalDebt")
        self.cash_and_cash_equivalents = self.data_extractor(
            variable=self.bs, date_required="cashAndCashEquivalents")
        self.inventory = self.data_extractor(
            variable=self.bs, date_required="inventory")
        self.accounts_receviables = self.data_extractor(
            variable=self.bs, date_required="netReceivables")
        self.accounts_payable = self.data_extractor(
            variable=self.bs, date_required="accountPayables")
        self.total_equity = self.data_extractor(
            variable=self.bs, date_required="totalEquity")
        self.total_assets = self.data_extractor(
            variable=self.bs, date_required="totalAssets")
        self.retained_earning = self.data_extractor(
            variable=self.bs, date_required="retainedEarnings")
        self.total_liabilities = self.data_extractor(
            variable=self.bs, date_required="totalLiabilities")

        # Converted to NumPy arrays to allow vector subtraction when computing Working Capital
        self.total_current_liabilities = np.array(
            self.data_extractor(
                variable=self.bs, date_required="totalCurrentLiabilities")
        )
        self.total_current_assets = np.array(
            self.data_extractor(
                variable=self.bs, date_required="totalCurrentAssets")
        )

    def cashflow(self) -> None:
        """Parses historical Cash Flow Statement metrics into internal 1D lists."""
        self.cash_from_operation = self.data_extractor(
            variable=self.cf, date_required="operatingCashFlow")
        self.capex = self.data_extractor(
            variable=self.cf, date_required="capitalExpenditure")
        self.credit_sales = self.data_extractor(
            variable=self.cf, date_required="accountsReceivables")

    def extras(self) -> None:
        """Extracts single-value snapshot metrics, market inputs, and valuation constants."""
        if not self.profile:
            raise AnalysisError("Company profile is empty. The ticker may be invalid.")
        if not self.rate:
            raise AnalysisError("Treasury yield data is empty. Try again later.")
        if not self.pnl:
            raise AnalysisError("Income statement is empty. The ticker may be invalid.")

        profile_row = self.profile[0] if isinstance(self.profile, list) else self.profile
        rate_row = self.rate[0] if isinstance(self.rate, list) else self.rate
        pnl_row = self.pnl[0] if isinstance(self.pnl, list) else self.pnl

        try:
            self.current_price = float(profile_row["price"])
            self.beta = float(profile_row["beta"])
            self.market_cap = float(profile_row["marketCap"])
            self.interest_rate = float(rate_row["year10"])
            self.dilluted_shares = float(pnl_row["weightedAverageShsOutDil"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                "Statements are missing price, beta, market cap, or share count."
            ) from exc

        if self.beta is None or self.dilluted_shares in (None, 0, 0.0):
            raise AnalysisError(
                "Missing beta or diluted shares. This ticker cannot be valued."
            )

        self.premium_rate, self.premium_warning = match_equity_risk_premium(
            self.premium, self.country
        )

    def pandas_dataframe(self) -> None:
        """Assembles extracted 1D vectors into structured Pandas DataFrames.

        Aligns financial metrics by fiscal year (`self.fy`) as the DataFrame index.
        Calculates derived metrics (e.g., Working Capital) during assembly.
        """
        self.df_pnl = pd.DataFrame(
            {
                "Revenue": self.revenue,
                "Ebit": self.net_income,
                "Interest Expenses": self.interest_expense,
                "Tax": self.tax,
                "Operating Income": self.operation_income,
                "Income Before Tax": self.income_before_tax,
                "Depreciation And Amortization": self.depreciation,
                "Net Income": self.net_income_,
                "Cost of Goods Sold": self.cogs,
            },
            index=self.fy,
        )

        self.df_bs = pd.DataFrame(
            {
                "Total Debt": self.total_debt,
                "Cash": self.cash_and_cash_equivalents,
                "Accounts Receivable": self.accounts_receviables,
                "Inventory": self.inventory,
                "Accounts Payable": self.accounts_payable,
                "Total Equity": self.total_equity,
                "Total Assets": self.total_assets,
                "Retained Earnings": self.retained_earning,
                "Working Capital": self.total_current_assets - self.total_current_liabilities,
                "Total Liabilities": self.total_liabilities,
            },
            index=self.fy,
        )

        self.df_cf = pd.DataFrame(
            {
                "Cashflow from operation": self.cash_from_operation,
                "Capex": self.capex,
                "Accounts Receivables": self.credit_sales,
            },
            index=self.fy,
        )

        self.df_extra = pd.DataFrame(
            {
                "Current Price": [self.current_price],
                "Beta": [self.beta],
                "Market Captilization": [self.market_cap],
                "Bond Interest Rate": [self.interest_rate],
                "Dilluted Shares": [self.dilluted_shares],
                "Equity Premium": [self.premium_rate],
            }
        )
