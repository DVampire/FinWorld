import aiohttp
import certifi
import ssl
import json
from datetime import datetime, timedelta

async def get_jsonparsed_data(request_url, timeout=60):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), timeout=timeout_obj) as session:
        async with session.get(request_url) as response:
            response.raise_for_status()
            text = await response.text()
            return json.loads(text)

def generate_intervals(start_date, end_date, interval_level='year', right_closed=False):
    intervals = []

    def right_endpoint(current, next):
        if right_closed:
            return next
        else:
            return next - timedelta(days=1)

    if interval_level == 'year':
        current_date = start_date
        while current_date < end_date:
            try:
                next_year = current_date.replace(year=current_date.year + 1)
            except ValueError:
                next_year = current_date.replace(month=3, day=1, year=current_date.year + 1)
            if next_year > end_date:
                next_year = end_date
            intervals.append((current_date, right_endpoint(current_date, next_year)))
            current_date = next_year
    elif interval_level == 'month':
        current_date = start_date
        while current_date < end_date:
            year, month = current_date.year, current_date.month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            if next_month > end_date:
                next_month = end_date
            intervals.append((current_date, right_endpoint(current_date, next_month)))
            current_date = next_month
    elif interval_level == 'day':
        current_date = start_date
        while current_date < end_date:
            next_day = current_date + timedelta(days=1)
            if next_day > end_date:
                next_day = end_date
            intervals.append((current_date, right_endpoint(current_date, next_day)))
            current_date = next_day
    else:
        return None

    return intervals