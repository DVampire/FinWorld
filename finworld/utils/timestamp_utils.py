import pandas as pd

def convert_timestamp_to_int(timestamp):
    return int(timestamp.timestamp())

def convert_int_to_timestamp(int_timestamp: int):
    return pd.Timestamp.fromtimestamp(int_timestamp)
