import pandas as pd
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
import os

# Set up API key
alpha_vantage_api_key = 'J6AAG38O90IQHXT4'

# List of the 100 most traded stock symbols (Example list)
most_traded_stocks = [
    'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'META', 'NVDA', 'PYPL', 'ADBE',
    'NFLX', 'INTC', 'CSCO', 'PFE', 'MRK', 'DIS', 'V', 'MA', 'JPM', 'BAC',
    'WMT', 'T', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'BA', 'NKE', 'MCD',
    'IBM', 'ORCL', 'ACN', 'AVGO', 'TXN', 'COST', 'QCOM', 'AMD', 'SPGI', 'AMT',
    'MDT', 'HON', 'LLY', 'UNP', 'UNH', 'UPS', 'NEE', 'AXP', 'LIN', 'LOW',
    'MS', 'GS', 'BLK', 'BK', 'TMO', 'ABBV', 'CVS', 'RTX', 'MMM', 'INTU',
    'SYK', 'CAT', 'CI', 'FIS', 'DHR', 'HUM', 'EL', 'LMT', 'ADP', 'SCHW',
    'ZTS', 'ISRG', 'GE', 'USB', 'FDX', 'ICE', 'SBUX', 'CME', 'NOW', 'TMUS',
    'GM', 'DE', 'MMC', 'ANTM', 'GILD', 'ADSK', 'PLD', 'ECL', 'APD', 'BSX',
    'MO', 'FISV', 'HCA', 'DG', 'F', 'CL', 'D', 'DUK', 'SO', 'ETN'
]

# Function to get historical stock prices from Yahoo Finance
def get_stock_prices(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="max")
    hist.reset_index(inplace=True)
    hist['Symbol'] = symbol
    return hist

# Function to get financial statements from Alpha Vantage
def get_financial_statements(symbol):
    fd = FundamentalData(key=alpha_vantage_api_key, output_format='pandas')
    income_statement, _ = fd.get_income_statement_annual(symbol)
    balance_sheet, _ = fd.get_balance_sheet_annual(symbol)
    cash_flow, _ = fd.get_cash_flow_annual(symbol)
    return income_statement, balance_sheet, cash_flow

# Function to get corporate actions (example: dividends from Yahoo Finance)
def get_corporate_actions(symbol):
    stock = yf.Ticker(symbol)
    dividends = stock.dividends
    dividends = dividends.reset_index()
    dividends['Symbol'] = symbol
    return dividends

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Loop over the most traded stocks
for symbol in most_traded_stocks:
    try:
        print(f"Processing {symbol}...")
        
        # Fetch data
        stock_prices = get_stock_prices(symbol)
        income_statement, balance_sheet, cash_flow = get_financial_statements(symbol)
        corporate_actions = get_corporate_actions(symbol)
        
        # Create directory for each company
        company_dir = f'data/{symbol}'
        if not os.path.exists(company_dir):
            os.makedirs(company_dir)
        
        # Save each table to a separate CSV file
        stock_prices.to_csv(f'{company_dir}/stock_prices.csv', index=False)
        income_statement.to_csv(f'{company_dir}/income_statement.csv', index=False)
        balance_sheet.to_csv(f'{company_dir}/balance_sheet.csv', index=False)
        cash_flow.to_csv(f'{company_dir}/cash_flow.csv', index=False)
        corporate_actions.to_csv(f'{company_dir}/corporate_actions.csv', index=False)
        
    except Exception as e:
        print(f"Failed to process {symbol}: {e}")
    
    break

print("Data collection complete.")
