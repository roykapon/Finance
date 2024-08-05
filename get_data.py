from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
import os

# Set up API key
alpha_vantage_api_key = "YSYKLOPDTQH07PLW"

# List of the 100 most traded stock symbols (Example list)
# fmt: off
most_traded_stocks = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "GOOG", "META", "NVDA", "PYPL", "ADBE", "NFLX", "INTC", "CSCO", "PFE", "MRK", "DIS", "V", "MA", "JPM", "BAC", "WMT", "T", "PG", "KO", "PEP", "XOM", "CVX", "BA", "NKE", "MCD", "IBM", "ORCL", "ACN", "AVGO", "TXN", "COST", "QCOM", "AMD", "SPGI", "AMT", "MDT", "HON", "LLY", "UNP", "UNH", "UPS", "NEE", "AXP", "LIN", "LOW", "MS", "GS", "BLK", "BK", "TMO", "ABBV", "CVS", "RTX", "MMM", "INTU", "SYK", "CAT", "CI", "FIS", "DHR", "HUM", "EL", "LMT", "ADP", "SCHW", "ZTS", "ISRG", "GE", "USB", "FDX", "ICE", "SBUX", "CME", "NOW", "TMUS", "GM", "DE", "MMC", "ANTM", "GILD", "ADSK", "PLD", "ECL", "APD", "BSX", "MO", "FISV", "HCA", "DG", "F", "CL", "D", "DUK", "SO", "ETN",]
# fmt: on

basedate = pd.Timestamp("0001-1-1")


def date_to_int(date):
    return (pd.Timestamp(date).to_pydatetime().replace(tzinfo=None) - basedate).days


def process_data(data: pd.DataFrame):
    data.rename(columns={"fiscalDateEnding": "Date"}, inplace=True)
    for col in data.columns:
        # convert all data to float and remove data of non-numeric type
        data = data.replace({"None": -1.0})
        data = data.convert_dtypes()

        if col == "reportedCurrency":
            if data[col].nunique() > 1:
                print(f"Multiple currencies: {data[col].unique()}")
                return None
            elif data[col].iloc[0] != "USD":
                print(f"Currency not USD: {data[col].iloc[0]}")
                return None

        if col == "Date":
            data[col] = data[col].apply(date_to_int)

        try:
            data[col] = data[col].astype(float)
        except:
            data.drop(columns=col, inplace=True)
    return data


# Function to get historical stock prices from Yahoo Finance
def get_stock_prices(symbol: str):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="max")
    hist.reset_index(inplace=True)
    hist["Symbol"] = symbol
    return hist


# Function to get financial statements from Alpha Vantage
def get_financial_statements(symbol: str):
    fd = FundamentalData(key=alpha_vantage_api_key, output_format="pandas")
    income_statement, _ = fd.get_income_statement_annual(symbol)
    balance_sheet, _ = fd.get_balance_sheet_annual(symbol)
    cash_flow, _ = fd.get_cash_flow_annual(symbol)
    return income_statement, balance_sheet, cash_flow


# Function to get corporate actions (example: dividends from Yahoo Finance)
def get_corporate_actions(symbol: str):
    stock = yf.Ticker(symbol)
    dividends = stock.dividends
    dividends = dividends.reset_index()
    dividends["Symbol"] = symbol
    return dividends


# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Loop over the most traded stocks
for symbol in most_traded_stocks:
    try:
        print(f"Processing {symbol}...")

        # Fetch data
        stock_prices = get_stock_prices(symbol)
        income_statement, balance_sheet, cash_flow = get_financial_statements(symbol)
        corporate_actions = get_corporate_actions(symbol)

        # Create directory for each company
        company_dir = f"data/{symbol}"
        if not os.path.exists(company_dir):
            os.makedirs(company_dir)

        data_dict = {"stock_prices": stock_prices, "income_statement": income_statement, "balance_sheet": balance_sheet, "cash_flow": cash_flow, "corporate_actions": corporate_actions}

        # process data
        for data_type, data in data_dict.items():
            data_dict[data_type] = process_data(data)

        # Save each table to a separate CSV file
        for data_type, data in data_dict.items():
            data.to_csv(f"{company_dir}/{data_type}.csv", index=False)

    except Exception as e:
        print(f"Failed to process {symbol}: {e}")


print("Data collection complete.")
