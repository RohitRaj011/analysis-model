import pandas as pd
import numpy as np
from typing import Any, List
# StoringAndCleaning is imported but not explicitly utilized in this snippet.
# If it handles preprocessing, ensure it modifies fetcher data before this class runs.
from analysis_model.data.StoringandCleaning import StoringAndCleaning
from analysis_model.data.Data_gathering import DataGathering


class Calculation:
    """Executes a Discounted Cash Flow (DCF) valuation pipeline using corporate financial data.

    This class orchestrates a sequence of dependent financial calculations, processing 
    historical data to establish growth rates, cost of capital (WACC), and ultimately 
    deriving the estimated Enterprise Value of a firm.

    Attributes:
        const_growth_rate (float): Perpetual growth rate for terminal value calculation (set to 2%).
        pnl (pd.DataFrame): Profit and Loss statements sorted chronologically.
        bs (pd.DataFrame): Balance Sheet statements sorted chronologically.
        cf (pd.DataFrame): Cash Flow statements sorted chronologically.
        extra (pd.DataFrame): Supplemental market metrics (Beta, Market Cap, etc.).
        index (List[int]): Financial/Fiscal years used as the chronological index.
        latest_year (int): The most recent fiscal year present in the dataset.
        data (pd.DataFrame): Aggregated historical metrics computed during execution.
        projections (pd.DataFrame): Projected 5-year financial metrics and cash flows.
        wacc (float): Computed Weighted Average Cost of Capital.
        terminal_value (float): Anticipated value of the company beyond the 5-year forecast window.
        terminal_value_discounted (float): Present value of the terminal value.
        sum_of_pv (float): Sum of the present values of the 5-year projected cash flows.
        enterprise_value (float): The final calculated intrinsic value of the operation.
    """

    def __init__(
        self,
        fetcher: StoringAndCleaning,
        const_growth_rate: float = 0.02,
        projection_years: int = 5,
    ) -> None:
        """Initializes the Calculation pipeline by extracting data from the fetcher module.

        Args:
            fetcher (StoringAndCleaning): Cleaned statement DataFrames.
            const_growth_rate: Perpetual growth rate for terminal value (default 2%).
            projection_years: Discrete forecast years (default 5).
        """
        # Base data frames must be assigned first and sorted chronologically (ascending)
        # to ensure that time-series analysis tools like .diff() and .rolling() evaluate accurately.
        self.const_growth_rate: float = const_growth_rate
        self.projection_years: int = int(projection_years)
        self.pnl: pd.DataFrame = fetcher.df_pnl.sort_index(ascending=True)
        self.bs: pd.DataFrame = fetcher.df_bs.sort_index(ascending=True)
        self.cf: pd.DataFrame = fetcher.df_cf.sort_index(ascending=True)
        self.extra: pd.DataFrame = fetcher.df_extra.sort_index(ascending=True)

        self.index: List[int] = fetcher.fy
        self.latest_year: int = max(self.index)
        self.data: pd.DataFrame = pd.DataFrame(
            index=self.index).sort_index(ascending=True)
        self.projections: pd.DataFrame = pd.DataFrame()

        # Explicit placeholders for attributes calculated downstream
        self.wacc: float = 0.0
        self.terminal_value: float = 0.0
        self.terminal_value_discounted: float = 0.0
        self.sum_of_pv: float = 0.0
        self.enterprise_value: float = 0.0

    def fetch_data(self) -> None:
        """Triggers the step-by-step execution of the financial model.

        Crucial: Order of execution matters entirely here. Downstream calculations 
        rely heavily on dataframes populated by upstream methods.
        """
        self.free_cashflow()
        self.tax_rate()
        self.Roic()
        self.reinvestment_rate()
        self.growth_rate()
        self.wacc_calc()
        self.five_year_projected_cashflow()
        self.discounted_value()
        self.terminal_value_calc()
        self.sum_of_pv_calc()
        self.enterprise_value_calc()

    def free_cashflow(self) -> None:
        """Calculates historical Free Cash Flow (FCF).

        FCF represents the cash a company generates after accounting for cash outflows 
        to support operations and maintain its capital assets (Capex).
        """
        # .abs() ensures Capex is treated subtractionally regardless of input sign convention.
        self.data["Free cashflow"] = self.cf["Cashflow from operation"] - \
            self.cf["Capex"].abs()

    def tax_rate(self) -> None:
        """Calculates the historical effective corporate tax rate.

        Required to determine the after-tax operating income for ROIC and WACC.
        """
        # .replace(0, np.nan) prevents ZeroDivisionError (resulting in Inf) if a company
        # reports zero income before tax in a given year.
        self.pnl["Tax rate"] = self.pnl["Tax"] / \
            self.pnl["Income Before Tax"].replace(0, np.nan)

    def Roic(self) -> None:
        """Calculates Return on Invested Capital (ROIC).

        Measures how efficiently a company allocates capital to profitable investments.
        Formula: NOPAT (Net Operating Profit After Tax) / Invested Capital.
        """
        inv_cap = self.bs["Total Debt"] + \
            self.bs["Total Equity"] - self.bs["Cash"]
        self.data["Roic"] = self.pnl["Operating Income"] * \
            (1 - self.pnl["Tax rate"]) / (inv_cap).replace(0, np.nan)

    def reinvestment_rate(self) -> None:
        """Calculates the Reinvestment Rate.

        Determines the percentage of after-tax operating earnings channeled back 
        into the business via Working Capital changes and Net Capital Expenditures.
        """
        self.data["Working Capital"] = self.bs["Accounts Receivable"] + \
            self.bs["Inventory"] - self.bs["Accounts Payable"]

        # .diff() computes the period-over-period change in working capital allocation.
        self.data["Change in Working capital"] = self.data["Working Capital"].diff()

        denominator = (self.pnl["Operating Income"]).replace(
            0, np.nan) * (1 - self.pnl["Tax rate"])
        self.data["Reinvestment Rate"] = (
            self.data["Change in Working capital"] +
            self.cf["Capex"].abs() - self.pnl["Depreciation And Amortization"]
        ) / denominator

    def growth_rate(self) -> None:
        """Calculates fundamental growth rate and its 3-year rolling average.

        Fundamental growth is driven by how much a company reinvests and the return 
        they achieve on those reinvestments (Reinvestment Rate * ROIC).
        """
        self.data["Growth Rate"] = self.data["Reinvestment Rate"] * \
            self.data["Roic"]

        # A 3-year rolling average smoothes out annual macroeconomic anomalies or
        # cyclical business spikes before projecting future numbers.
        self.data["Average Growth Rate"] = self.data["Growth Rate"].rolling(
            window=3).mean()

    def wacc_calc(self) -> None:
        """Computes the Weighted Average Cost of Capital (WACC) for the latest fiscal year.

        WACC serves as the discount rate for future cash flows, acting as the hurdle 
        rate reflecting risk across both debt and equity allocations.
        """
        self.data["Total Value of Capital"] = self.bs["Total Debt"] + \
            self.bs["Total Equity"]

        # 2-period rolling mean calculates average debt over the year to match up with
        # annualized interest expenses cleanly.
        self.data["Average Borrowings"] = self.bs["Total Debt"].rolling(
            window=2).mean()

        market_value_of_equity = self.extra.loc[0, "Market Captilization"]
        market_value_of_debt = self.bs.loc[self.latest_year, "Total Debt"]
        total_value_of_capital = market_value_of_equity + market_value_of_debt

        average_debt = self.data.loc[self.latest_year, "Average Borrowings"]
        interest_expenses = self.pnl.loc[self.latest_year, "Interest Expenses"]
        cost_of_debt = interest_expenses / average_debt if average_debt != 0 else 0

        # Capital Asset Pricing Model (CAPM) formula for Cost of Equity: RiskFree + (Beta * EquityPremium)
        cost_of_equity = (self.extra.loc[0, "Bond Interest Rate"] + (
            self.extra.loc[0, "Beta"] * self.extra.loc[0, "Equity Premium"]
        ))/100
        tax_rate = self.pnl.loc[self.latest_year, "Tax rate"]

        # Blended WACC calculation (Debt is tax-shielded, Equity is not)
        self.wacc = (market_value_of_equity / total_value_of_capital * cost_of_equity) + \
                    ((market_value_of_debt / total_value_of_capital *
                     cost_of_debt) * (1 - tax_rate))

    def five_year_projected_cashflow(self) -> None:
        """Projects Free Cash Flows 5 years into the future.

        Applies the compounding calculated historical rolling average growth rate 
        against the latest year's actual Free Cash Flow.
        """
        self.projections = pd.DataFrame(index=range(1, self.projection_years + 1))
        self.projections["Growth Rate"] = 1 + \
            self.data.loc[self.latest_year, "Average Growth Rate"]

        # .cumprod() models the exponential compounding effects over the 5-year horizon.
        self.free_cashflow_to_firm = (self.pnl.loc[self.latest_year, "Ebit"] * (1 - self.pnl.loc[self.latest_year, "Tax rate"])) + self.pnl.loc[self.latest_year,
                                                                                                                                                "Depreciation And Amortization"] - abs(self.cf.loc[self.latest_year, "Capex"]) - self.data.loc[self.latest_year, "Change in Working capital"]
        self.projections["Projected Cashflow"] = self.free_cashflow_to_firm * \
            self.projections["Growth Rate"].cumprod()

    def discounted_value(self) -> None:
        """Applies the discount factor to projected cash flows.

        Translates future cash value into 'today's dollars' based on the computed WACC.
        """
        # Time value of money: PV = FV / (1 + WACC)^t
        self.projections["Discount Value"] = 1 / \
            ((1 + self.wacc) ** self.projections.index)
        self.projections["Cashflow_after_discount"] = self.projections["Projected Cashflow"] * \
            self.projections["Discount Value"]

    def terminal_value_calc(self) -> None:
        """Calculates the Terminal Value using the Gordon Growth Model.

        Accounts for the estimated cash flows the firm generates in perpetuity 
        beyond the 5-year discrete projection window.
        """
        # Gordon Growth formula: (Final Year Cashflow * (1 + Perpetual Growth Rate)) / (WACC - Perpetual Growth Rate)
        last_year = self.projection_years
        self.terminal_value = (self.projections.loc[last_year, "Projected Cashflow"]) * (1 + self.const_growth_rate) / \
                              (self.wacc - self.const_growth_rate)
        # Discount the massive lump-sum perpetual value back to present value terms
        self.terminal_value_discounted = self.terminal_value / \
            (1 + self.wacc) ** last_year

    def sum_of_pv_calc(self) -> None:
        """Aggregates the present value of the 5-year discrete projection window."""
        self.sum_of_pv = self.projections["Cashflow_after_discount"].sum()

    def enterprise_value_calc(self) -> None:
        """Computes total Enterprise Value by combining short-term and perpetual cash flows."""
        self.enterprise_value = self.sum_of_pv + self.terminal_value_discounted


if __name__ == "__main__":
    data_fetcher = DataGathering("AAPL")
    cleaned = StoringAndCleaning(data_fetch=data_fetcher)
    cleaned.fetch_dataframe()
    cal = Calculation(fetcher=cleaned)
    cal.fetch_data()
    print(cal.enterprise_value)
    pass
