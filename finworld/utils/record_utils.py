import numpy as np
from typing import Dict, Any
import pandas as pd

class Records():
    def __init__(self):
        self.data = dict()

    def add(self, item: Dict[str, Any]):
        """
        Add a new record to the records.
        :param item: A dictionary containing the record information.
        """
        for key, value in item.items():
            self.data.setdefault(key, []).append(value)

    def avg(self):
        """
        Calculate the average of each record.
        :return: A dictionary containing the average of each record.
        """
        return {key: np.mean(value) for key, value in self.data.items()}

class TradingRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            action_label = [],
            ret = [],
            total_profit=[],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the trading records.
        :param info: A dictionary containing the trading information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the trading records to a pandas DataFrame.
        :return: A pandas DataFrame containing the trading records.
        """
        df = pd.DataFrame(self.data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df

class PortfolioRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            ret = [],
            total_profit = [],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the portfolio records.
        :param info: A dictionary containing the portfolio information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the portfolio records to a pandas DataFrame.
        :return: A pandas DataFrame containing the portfolio records.
        """
        df = pd.DataFrame(self.data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df