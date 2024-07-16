from datetime import datetime, timedelta
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

# Global variable for the number of days to average
DAYS_TO_AVERAGE = 5

def int_to_date(date):
    epoch = datetime(1, 1, 1)  # January 1st, year 0
    target_date = epoch + timedelta(days=int(date))
    return target_date.strftime('%Y-%m-%d')

class StockDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data()

    def __len__(self):
        return len(self.data["stock_prices"])

    def __getitem__(self, idx):
        return self.data["stock_prices"].iloc[idx]

    def load_data(self):
        data = {}

        # Load each data type into the dictionary
        for data_type in ["stock_prices", "income_statement", "balance_sheet", "cash_flow", "corporate_actions"]:
            data[data_type] = pd.read_csv(f"{self.data_dir}\\{data_type}.csv")
        return data

    def sample_data_up_to_date(self):
        # Sample a random date within the available range of stock prices
        stock_prices = self.data["stock_prices"]
        # sampled_date_index = random.randint(0, len(stock_prices) - 1)
        sampled_date_index = random.randint(int((len(stock_prices) - 1)*0.75), len(stock_prices) - 1)
        sampled_date = stock_prices.loc[sampled_date_index, "Date"]

        # Generate dictionary up to, but not including, the sampled date for each data type
        data_up_to_date = {}
        for data_type, df in self.data.items():
            data_up_to_date[data_type] = df[df["Date"] < sampled_date]

        # Calculate average stock value over the next DAYS_TO_AVERAGE days
        next_days_prices = stock_prices[(stock_prices["Date"] > sampled_date) & (stock_prices["Date"] <= sampled_date + 5)]
        average_stock_value = next_days_prices["Close"].mean()

        return data_up_to_date, average_stock_value, sampled_date


# Example usage:
if __name__ == "__main__":
    data_dir = "data\\AAPL"  # Example directory, replace with your data directory path
    dataset = StockDataset(data_dir)

    # Example of loading data up to, but not including, a sampled date and averaging the next 5 days
    for _ in range(5):
        sampled_data, average_stock_value, sampled_date = dataset.sample_data_up_to_date()
        print("Sampled Data:")
        for data_type, df in sampled_data.items():
            print(f"\n{data_type}:\n{df.head()}")

        print(f"\nAverage Stock Value over the Next {DAYS_TO_AVERAGE} Days: {average_stock_value}, Day: {int_to_date(sampled_date)}\n")
