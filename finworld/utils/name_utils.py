import hashlib
from typing import Optional

def get_tag_name(tag:Optional[str] = None,
                 assets_name: Optional[str] = None,
                 source: Optional[str] = None,
                 data_type: Optional[str] = None,
                 level: Optional[str] = None) -> str:
    """
    Generate a tag name based on the asset name, source, and data type.

    :param assets_name: The name of the asset (e.g., 'dj30').
    :param source: The source of the data (e.g., 'fmp', 'yahoo').
    :param data_type: The type of data (e.g., 'price', 'fundamental').
    :return: A formatted tag name.
    """
    # e.g., 'dj30_fmp_price_1day'

    filters = [assets_name, source, data_type, level]
    name = "_".join([str(f) for f in filters if f is not None])
    return name if tag is None else tag

def get_newspage_name(symbol: str, timestamp: str, title: str) -> str:
    """
    Generate a news page name based on the symbol and source.

    :param symbol: The stock symbol (e.g., 'AAPL').
    :param source: The source of the news (default is 'fmp').
    :return: A formatted news page name.
    """
    return hashlib.md5(f'{symbol} {timestamp} {title}'.encode()).hexdigest()