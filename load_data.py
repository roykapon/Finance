from datetime import datetime, timedelta
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Global variable for the number of days to average
DAYS_TO_AVERAGE = 5
DATA_TYPES = ["stock_prices", "income_statement", "balance_sheet", "cash_flow", "corporate_actions"]
STOCK_PRICES_ID = DATA_TYPES.index("stock_prices")
WINDOW_SIZE = 365 * 3  # 3 years of data
WINDOW_INTERVAL = 5  # 5 days

def stock_collate(batch):
    inputs, targets, dates = zip(*batch)

    # Separate stock and balance features
    padded = [pad_sequence(input[i] for input in inputs) for i in range(len(inputs[0]))]
    return padded, torch.FloatTensor(targets), torch.FloatTensor(dates)


def int_to_date(date):
    epoch = datetime(1, 1, 1)  # January 1st, year 0
    target_date = epoch + timedelta(days=int(date))
    return target_date.strftime("%Y-%m-%d")


class StockDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = self.load_data()
        self.input_sizes = [df.shape[1] for df in self.data]

    def load_data(self) -> list[pd.DataFrame]:
        # Load each data type into a list
        data = [pd.read_csv(f"{self.data_dir}\\{data_type}.csv") for data_type in DATA_TYPES]
        data[STOCK_PRICES_ID] = data[STOCK_PRICES_ID][(data[STOCK_PRICES_ID]["Date"] % WINDOW_INTERVAL == 0)]
        return data

    def __len__(self):
        return len(self.data[STOCK_PRICES_ID])

    def __getitem__(self, idx):
        # Sample a random date within the available range of stock prices
        stock_prices = self.data[STOCK_PRICES_ID]
        # sampled_date_index = random.randint(0, len(stock_prices) - 1)
        sampled_date_index = random.randint(int((len(stock_prices) - 1) * 0.75), len(stock_prices) - 1)
        sampled_date: float = stock_prices.iloc[sampled_date_index, 0]

        # Generate dictionary up to, but not including, the sampled date for each data type
        data_up_to_date = [torch.tensor(df[(df["Date"] < sampled_date) & (df["Date"] < sampled_date - WINDOW_SIZE)].values) for df in self.data]

        # Calculate average stock value over the next WINDOW_SIZE days
        next_days_prices = stock_prices[(stock_prices["Date"] > sampled_date) & (stock_prices["Date"] <= sampled_date + WINDOW_SIZE)]
        average_stock_value = next_days_prices["Close"].mean()

        return data_up_to_date, average_stock_value, sampled_date
