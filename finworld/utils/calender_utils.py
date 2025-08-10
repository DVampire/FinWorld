import pandas as pd
from datetime import datetime
from typing import Optional, Any
import enum
from collections import OrderedDict

class TimeLevel(enum.Enum):
    """
    Enum for time levels.
    """
    DAY = "1day"
    HOUR = "1hour"
    MINUTE = "1min"
    SECOND = "1sec"

    @classmethod
    def from_string(cls, level: str) -> 'TimeLevel':
        """
        Convert a string to a TimeLevel enum.
        """
        level = level.lower()
        if level == "1day":
            return cls.DAY
        elif level == "1hour":
            return cls.HOUR
        elif level == "1min":
            return cls.MINUTE
        elif level == "1sec":
            return cls.SECOND

class TimeLevelFormat(enum.Enum):
    """
    Enum for time level formats.
    """
    DAY = "%Y-%m-%d"
    HOUR = "%Y-%m-%d %H"
    MINUTE = "%Y-%m-%d %H:%M"
    SECOND = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def from_string(cls, level: str) -> 'TimeLevelFormat':
        """
        Convert a string to a TimeLevelFormat enum.
        """
        level = level.lower()
        if level == "1day":
            return cls.DAY
        elif level == "1hour":
            return cls.HOUR
        elif level == "1min":
            return cls.MINUTE
        elif level == "1sec":
            return cls.SECOND

def get_start_end_timestamp(
    start_timestamp: Optional[Any] = None,
    end_timestamp: Optional[Any] = None,
    level: TimeLevel = TimeLevel.DAY,
):
    start_timestamp = pd.to_datetime(start_timestamp)
    end_timestamp = pd.to_datetime(end_timestamp)

    if level == TimeLevel.DAY:
        start_timestamp = start_timestamp
        end_timestamp = end_timestamp
        end_timestamp = end_timestamp + pd.DateOffset(days=1) - pd.DateOffset(seconds=1)
    elif level == TimeLevel.HOUR:
        start_timestamp = start_timestamp
        end_timestamp = end_timestamp + pd.DateOffset(hours=1) - pd.DateOffset(seconds=1)
    elif level == TimeLevel.MINUTE:
        start_timestamp = start_timestamp
        end_timestamp = end_timestamp
        end_timestamp = end_timestamp + pd.DateOffset(minutes=1) - pd.DateOffset(seconds=1)
    elif level == TimeLevel.SECOND:
        start_timestamp = start_timestamp
        end_timestamp = end_timestamp

    return start_timestamp, end_timestamp

def calculate_time_info(start_timestamp: str,
                        end_timestamp: str,
                        level: TimeLevel):

    """Calculate the number of embeddings based on the time level."""
    start_timestamp = pd.to_datetime(start_timestamp)
    end_timestamp = pd.to_datetime(end_timestamp)

    num_years = len(range(start_timestamp.year, end_timestamp.year + 1))
    min_year = 0
    num_months = 12  # January to December
    min_month = 1  # January
    num_weekdays = 7  # Monday to Sunday
    min_weekday = 0  # Monday
    num_days = 32  # 1 to 31 days in a month
    min_day = 1  # 1st day of the month

    if level == TimeLevel.DAY:
        columns = ["day", "month", "weekday", "year"]  # sorted
        embedding_info = OrderedDict({
            "day": {"length": num_days, "min": min_day},
            "month": {"length": num_months, "min": min_month},
            "weekday": {"length": num_weekdays, "min": min_weekday},
            "year": {"length": num_years, "min": min_year}
        })
    elif level == TimeLevel.HOUR:
        num_hours = 24
        min_hours = 0
        columns = ["day", "hour", "month", "weekday", "year"]  # sorted
        embedding_info = OrderedDict({
            "day": {"length": num_days, "min": min_day},
            "hour": {"length": num_hours, "min": min_hours},
            "month": {"length": num_months, "min": min_month},
            "weekday": {"length": num_weekdays, "min": min_weekday},
            "year": {"length": num_years, "min": min_year}
        })
    elif level == TimeLevel.MINUTE:
        num_hours = 24
        min_hours = 0
        num_minutes = 60
        min_minutes = 0
        columns = ["day", "hour", "minute", "month", "weekday", "year"]
        embedding_info = OrderedDict({
            "day": {"length": num_days, "min": min_day},
            "hour": {"length": num_hours, "min": min_hours},
            "minute": {"length": num_minutes, "min": min_minutes},
            "month": {"length": num_months, "min": min_month},
            "weekday": {"length": num_weekdays, "min": min_weekday},
            "year": {"length": num_years, "min": min_year}
        })
    elif level == TimeLevel.SECOND:
        num_hours = 24
        min_hours = 0
        num_minutes = 60
        min_minutes = 0
        num_seconds = 60
        min_seconds = 0
        columns = ["day", "hour", "minute", "month", "second", "weekday", "year"]
        embedding_info = OrderedDict({
            "day": {"length": num_days, "min": min_day},
            "hour": {"length": num_hours, "min": min_hours},
            "minute": {"length": num_minutes, "min": min_minutes},
            "month": {"length": num_months, "min": min_month},
            "second": {"length": num_seconds, "min": min_seconds},
            "weekday": {"length": num_weekdays, "min": min_weekday},
            "year": {"length": num_years, "min": min_year}
        })
    else:
        raise ValueError(f"Unsupported time level: {level}")

    res_info = dict(
        embedding_info=embedding_info,
        columns=columns
    )

    return res_info

