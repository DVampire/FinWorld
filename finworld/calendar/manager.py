from typing import Optional, Union, Dict, Any, List

import pandas as pd

from finworld.calendar.mapper import codemapper
from finworld.log import logger

class CalendarManager():
    """
    A manager class for handling financial calendars.

    This class provides methods to retrieve and manage financial calendars based on exchange codes.
    It uses a codemapper to map exchange codes to calendar objects.
    """
    def get_code(self, symbol_info: Dict[str, Any]) -> str:
        """
        Get the stock code for a given symbol information.

        :param symbol_info: Dictionary containing symbol information including exchange.
        :return: Corresponding stock code.
        """
        if "exchange" in symbol_info:
            exchange = symbol_info["exchange"]
        else:
            logger.warning("Exchange not found in symbol_info, using default 'New York Stock Exchange'.")
            exchange = "New York Stock Exchange"

        code = codemapper.get_code(exchange)

        return code

    def get_valid_days(self,
                       symbol_info: Dict[str, Any],
                       start_date: Union[str, Any],
                       end_date: Union[str, Any]
                       ):
        """
        Get valid trading days for a given symbol within a date range.

        :param symbol_info: Dictionary containing symbol information including exchange.
        :param start_date: Start date for the query.
        :param end_date: End date for the query.
        :return: List of valid trading days.
        """
        if "exchange" in symbol_info:
            exchange = symbol_info["exchange"]
        else:
            logger.warning("Exchange not found in symbol_info, using default 'New York Stock Exchange'.")
            exchange = "New York Stock Exchange"

        valid_days = codemapper.get_valid_days(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )

        return valid_days

    def get_num_periods(self,
                        symbol_info: Dict[str, Any],
                        level: str) -> int:
        """
        Get the number of periods for a given time level.

        :param symbol_info: Dictionary containing symbol information including exchange.
        :param level: Time level for which the number of periods is requested.
        :return: Number of periods for the specified time level.
        """
        if "exchange" in symbol_info:
            exchange = symbol_info["exchange"]
        else:
            logger.warning("Exchange not found in symbol_info, using default 'New York Stock Exchange'.")
            exchange = "New York Stock Exchange"

        return codemapper.get_num_periods(exchange, level)

    def convert_timezone(self,
                         symbol_info: Dict[str, Any],
                         dataframe: pd.DataFrame,
                         ) -> pd.DataFrame:
        """
        Convert a date from one timezone to another.

        :param symbol_info: Dictionary containing symbol information including exchange.
        :param date: Date to be converted.
        :param from_tz: Source timezone.
        :param to_tz: Target timezone.
        :return: Converted date in the target timezone.
        """
        if "exchange" in symbol_info:
            exchange = symbol_info["exchange"]
        else:
            logger.warning("Exchange not found in symbol_info, using default 'New York Stock Exchange'.")
            exchange = "New York Stock Exchange"

        return codemapper.convert_timezone(exchange, dataframe)

calendar_manager = CalendarManager()

if __name__ == '__main__':
    # Example usage
    from pandas_market_calendars import get_calendar
    import akshare as ak

    manager = CalendarManager()
    symbol_info = {"exchange": "New York Stock Exchange"}
    code = manager.get_code(symbol_info)
    print(f"Code for the exchange {symbol_info['exchange']}: {code}")

    valid_days = manager.get_valid_days(symbol_info, "2023-01-01", "2023-01-31")
    print(f"Valid trading days for {symbol_info['exchange']} from 2023-01-01 to 2023-01-31: {valid_days}")

    num_periods = manager.get_num_periods(symbol_info, "1day")
    print(f"Number of periods for {symbol_info['exchange']} at '1day' level: {num_periods}")