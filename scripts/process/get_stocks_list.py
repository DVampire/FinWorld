import os

import pandas as pd
from dotenv import load_dotenv
from urllib.request import urlopen
import certifi
import json
load_dotenv(verbose=True)
from glob import glob
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count, Manager
import numpy as np

from finworld.utils import assemble_project_path

FMP_API_KEY = os.environ.get('FMP_API_KEY')

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def download_symbol(chunk):
    results = []
    bar_format = f"Download:" + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"
    for (symbol, info) in tqdm(chunk, bar_format=bar_format):
        url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}'
        symbol_info = get_jsonparsed_data(url)[0]
        results.append(symbol_info)
    return results

def wrapper(chunk, results):
    result = download_symbol(chunk)
    results.extend(result)

def download_full():
    url = f'https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}'

    data = get_jsonparsed_data(url)

    num = len(data)
    name = f'full{int(num / 1000)}k'

    with open(assemble_project_path(f'configs/_asset_list_/{name}.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def download_assets(assets_name):

    with open(assemble_project_path(f'configs/_asset_list_/{assets_name}.json'), "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.items())
    elif isinstance(data, list):
        data = {
            item['symbol']: item for item in data
        }
        data = list(data.items())

    num_processes = min(cpu_count(), 4)
    download_chunks = np.array_split(data, num_processes, axis=0)

    manager = Manager()
    results = manager.list()

    with Pool(processes=num_processes) as pool:
        pool.starmap(wrapper, [(chunk, results) for chunk in download_chunks])

    final_data = list(results)
    final_data = sorted(final_data, key=lambda x: x['symbol'])
    final_data = {
        item['symbol']: item for item in final_data
    }

    with open(assemble_project_path(f'configs/_asset_list_/{assets_name}.json'), "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # download_full()
    # download_assets('dj30')
    # download_assets('exp')
    # download_assets('hs300')
    # download_assets('sp500')
    # download_assets('nasdaq100')
    download_assets('sse50')