from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from copy import deepcopy

from finworld.registry import FACTOR
from finworld.log import logger

EPS = 1e-12

@FACTOR.register_module('Alpha158')
class Alpha158:
    """
    Alpha158 is a collection of 158 alpha factors used in quantitative finance.
    Each factor is designed to capture different aspects of stock price movements.
    refer to: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
    """

    def __init__(self, windows: List[int] = None, level: str = None):

        self.windows: List[int] = windows if windows is not None else [5, 10, 20, 30, 60]
        self.level = level if level is not None else '1day'

        self.names: List[str] = []

        self.names.extend([
            'kmid', 'kmid2', 'klen', 'kup', 'kup2', 'klow', 'klow2', 'ksft', 'ksft2',
        ])
        for window in windows or windows:
            self.names.extend([
                f'roc_{window}', f'ma_{window}', f'std_{window}', f'beta_{window}',
                f'max_{window}', f'min_{window}', f'qtlu_{window}', f'qtld_{window}',
                f'rank_{window}', f'imax_{window}', f'imin_{window}', f'imxd_{window}',
                f'rsv_{window}', f'cntp_{window}', f'cntn_{window}', f'cntd_{window}',
                f'corr_{window}', f'cord_{window}', f'sump_{window}', f'sumn_{window}',
                f'sumd_{window}', f'vma_{window}', f'vstd_{window}',
                f'wvma_{window}', f'vsump_{window}', f'vsumn_{window}', f'vsumd_{window}'
            ])
        self.names.append('logvol')
        self.names = list(sorted(self.names))

    async def _kmid(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        K-Mid price.
        """
        df = deepcopy(df)
        df['kmid'] = (df['close'] - df['open']) / (df['close'] + EPS)
        df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + EPS)
        factors_info = {
            'kmid': '(close - open) / close',
            'kmid2': '(close - open) / (high - low)'
        }
        df = df[['kmid', 'kmid2']].copy()
        return df, factors_info

    async def _klen(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        K-Length price.
        """
        df['klen'] = (df['high'] - df['low']) / (df['open'] + EPS)
        factors_info = {
            'klen': '(high - low) / open'
        }
        df = df[['klen']].copy()
        return df, factors_info

    async def _kup(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        K-Up price.
        """
        if 'max_oc' not in df.columns:
            df['max_oc'] = df[['open', 'close']].max(axis=1)
        df['kup'] = (df['high'] - df['max_oc']) / (df['open'] + EPS)
        df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + EPS)
        factors_info = {
            'kup': '(high - max(open, close)) / open',
            'kup2': '(high - max(open, close)) / (high - low)'
        }
        df = df[['kup', 'kup2']].copy()
        return df, factors_info

    async def _klow(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        K-Low price.
        """
        if 'min_oc' not in df.columns:
            df['min_oc'] = df[['open', 'close']].min(axis=1)
        df['klow'] = (df['min_oc'] - df['low']) / (df['open'] + EPS)
        df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + EPS)
        factors_info = {
            'klow': '(min(open, close) - low) / open',
            'klow2': '(min(open, close) - low) / (high - low)'
        }
        df = df[['klow', 'klow2']].copy()
        return df, factors_info

    async def _ksft(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        K-Shift price.
        """
        df = deepcopy(df)
        df['ksft'] = (2 * df['close'] - df['high'] - df['low']) / (df['open'] + EPS)
        df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + EPS)

        factors_info = {
            'ksft': '(2 * close - high - low) / open',
            'ksft2': '(2 * close - high - low) / (high - low)'
        }
        df = df[['ksft', 'ksft2']].copy()
        return df, factors_info

    async def _roc(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'roc_{window}'
            df[col_name] = df['close'].shift(window) / (df['close'] + EPS)
            factors_info[col_name] = f'close.shift({window}) / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _ma(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'ma_{window}'
            df[col_name] = df['close'].rolling(window=window).mean() / (df['close'] + EPS)
            factors_info[col_name] = f'close.rolling({window}).mean() / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _std(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'std_{window}'
            df[col_name] = df['close'].rolling(window=window).std() / (df['close'] + EPS)
            factors_info[col_name] = f'close.rolling({window}).std() / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _beta(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'beta_{window}'
            df[col_name] = (df['close'].shift(window) - df['close']) / (window * df['close'] + EPS)
            factors_info[col_name] = f'(close.shift({window}) - close) / ({window} * close)'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _max(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'max_{window}'
            df[col_name] = df['close'].rolling(window=window).max() / (df['close'] + EPS)
            factors_info[col_name] = f'close.rolling({window}).max() / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _min(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'min_{window}'
            df[col_name] = df['close'].rolling(window=window).min() / (df['close'] + EPS)
            factors_info[col_name] = f'close.rolling({window}).min() / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _qtlu(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'qtlu_{window}'
            df[col_name] = (df['close'] - df['close'].rolling(window=window).quantile(0.8)) / (df['close'] + EPS)
            factors_info[col_name] = f'(close - close.rolling({window}).quantile(0.8)) / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _qtld(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'qtld_{window}'
            df[col_name] = (df['close'] - df['close'].rolling(window=window).quantile(0.2)) / (df['close'] + EPS)
            factors_info[col_name] = f'(close - close.rolling({window}).quantile(0.2)) / close'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _rank(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'rank_{window}'
            df[col_name] = df['close'].rolling(window=window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False) / window
            factors_info[col_name] = f'close.rolling({window}).apply(lambda x: x.rank(pct=True).iloc[-1])'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _imax(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'imax_{window}'
            df[col_name] = df['high'].rolling(window=window).apply(np.argmax) / window
            factors_info[col_name] = f'high.rolling({window}).max() / window'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _imin(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'imin_{window}'
            df[col_name] = df['low'].rolling(window=window).apply(np.argmin) / window
            factors_info[col_name] = f'low.rolling({window}).min() / window'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _imxd(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'imxd_{window}'
            df[col_name] = (df['high'].rolling(window=window).apply(np.argmax) - df['low'].rolling(window=window).apply(np.argmin)) / window
            factors_info[col_name] = f'(high.rolling({window}).argmax() - low.rolling({window}).argmin()) / window'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _rsv(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'rsv_{window}'

            shift = df['close'].shift(window)
            min = df['low'].where(df['low'] < shift, shift)
            max = df['high'].where(df['high'] > shift, shift)

            df[col_name] = (df['close'] - min) / (max - min + EPS)
            factors_info[col_name] = f'(close - min(low, close.shift({window}))) / (max(high, close.shift({window})) - min(low, close.shift({window})) + EPS)'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info


    async def _cnt(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Count of positive returns over a rolling window.
        """
        df = deepcopy(df)

        df['ret1'] = df['close'].pct_change(1)

        factors_info = {}
        for window in windows:
            col_name = f'cntp_{window}'
            df[col_name] = df['ret1'].gt(0).rolling(window=window).sum() / window
            factors_info[col_name] = f'ret1.gt(0).rolling({window}).sum() / {window}'

            col_name = f'cntn_{window}'
            df[col_name] = df['ret1'].lt(0).rolling(window=window).sum() / window
            factors_info[col_name] = f'ret1.lt(0).rolling({window}).sum() / {window}'

            col_name = f'cntd_{window}'
            df[col_name] = df[f'cntp_{window}'] - df[f'cntn_{window}']
            factors_info[col_name] = f'{col_name} - {f"cntn_{window}"}'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _corr(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Correlation of close prices with a rolling window.
        """
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'corr_{window}'
            df1 = df['close'].rolling(window=window)
            df2 = np.log(df["volume"] + 1).rolling(window=window)
            df[col_name] = df1.corr(pairwise=df2)
            factors_info[col_name] = f'close.rolling({window}).corr(np.log(volume + 1).rolling({window}))'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _cord(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Correlation of close prices with a rolling window, normalized.
        """
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'cord_{window}'

            df1 = df['close']
            df_shift1 = df1.shift(1)
            df2 = df['volume']
            df_shift2 = df2.shift(1)
            df1 = df1 / (df_shift1 + EPS)
            df2 = np.log(df2 / (df_shift2 + EPS) + 1)
            df[col_name] = df1.rolling(window=window).corr(pairwise=df2.rolling(window=window))

            factors_info[col_name] = f'(close / close.shift(1) + EPS).rolling({window}).corr(np.log(volume / volume.shift(1) + 1).rolling({window}))'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _sum(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Sum of positive returns over a rolling window.
        """
        df = deepcopy(df)

        df['ret1'] = df['close'].pct_change(1)
        df['abs_ret1'] = df['ret1'].abs()
        df['pos_ret1'] = df['ret1']
        df['pos_ret1'] = np.where(df['pos_ret1'] < 0, 0, df['pos_ret1'])

        factors_info = {}
        for window in windows:
            col_name = f'sump_{window}'
            df[col_name] = df['pos_ret1'].rolling(window=window).sum() / (df['abs_ret1'].rolling(window=window).sum() + EPS)
            factors_info[col_name] = f'pos_ret1.rolling({window}).sum() / abs_ret1.rolling({window}).sum()'

            col_name = f'sumn_{window}'
            df[col_name] = 1.0 - df[f'sump_{window}']
            factors_info[col_name] = f'1 - {col_name}'

            col_name = f'sumd_{window}'
            df[col_name] = 2.0 * df[f'sump_{window}'] - 1.0
            factors_info[col_name] = f'2 * {col_name} - 1'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _vma(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Volume-weighted moving average.
        """
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'vma_{window}'
            df[col_name] = df['volume'].rolling(window=window).mean() / (df['volume'] + EPS)
            factors_info[col_name] = f'volume.rolling({window}).mean() / volume'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _vstd(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Volume-weighted standard deviation.
        """
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'vstd_{window}'
            df[col_name] = df['volume'].rolling(window=window).std() / (df['volume'] + EPS)
            factors_info[col_name] = f'volume.rolling({window}).std() / volume'
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _wvma(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Weighted volume moving average.
        """
        df = deepcopy(df)
        factors_info = {}
        for window in windows:
            col_name = f'wvma_{window}'

            shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
            df1 = shift.rolling(window).std()
            df2 = shift.rolling(window).mean()

            df[col_name] = (df1 / (df2 + EPS)).fillna(0)
            factors_info[col_name] = f'abs(close / close.shift(1) - 1).rolling({window}).std() / abs(close / close.shift(1) - 1).rolling({window}).mean()'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _vsum(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Volume sum over a rolling window.
        """
        df = deepcopy(df)
        df['vchg1'] = df['volume'] - df['volume'].shift(1)
        df['abs_vchg1'] = np.abs(df['vchg1'])
        df['pos_vchg1'] = df['vchg1']
        df['pos_vchg1'] = np.where(df['pos_vchg1'] < 0, 0, df['pos_vchg1'])

        factors_info = {}
        for window in windows:
            col_name = f'vsump_{window}'
            df[col_name] = df['pos_vchg1'].rolling(window=window).sum() / (df['abs_vchg1'].rolling(window=window).sum() + EPS)
            factors_info[col_name] = f'pos_vchg1.rolling({window}).sum() / abs_vchg1.rolling({window}).sum()'

            col_name = f'vsumn_{window}'
            df[col_name] = 1.0 - df[f'vsump_{window}']
            factors_info[col_name] = f'1 - {col_name}'

            col_name = f'vsumd_{window}'
            df[col_name] = 2.0 * df[f'vsump_{window}'] - 1.0
            factors_info[col_name] = f'2 * {col_name} - 1'

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _logvol(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """
        Logarithm of volume.
        """
        df = deepcopy(df)
        df['logvol'] = np.log(df['volume'] + 1)
        factors_info = {
            'logvol': 'log(volume + 1)'
        }
        df = df[['logvol']].copy()
        return df, factors_info

    async def run(self, df: DataFrame,  windows: List[int] = None, level: str = None) -> Dict[str, Any]:
        """
        Run the specified alpha factor on the given DataFrame.

        :param df: Input DataFrame containing stock data.
        :param factor_name: Name of the alpha factor to compute.
        :param windows: List of window sizes for rolling calculations (if applicable).
        :param level: Level of detail for time-based factors.
        :return: Tuple of DataFrame with computed factors and a dictionary with factor descriptions.
        """
        assert 'timestamp' in df.columns, "DataFrame must contain a 'timestamp' column."
        assert 'open' in df.columns, "DataFrame must contain an 'open' column."
        assert 'high' in df.columns, "DataFrame must contain a 'high' column."
        assert 'low' in df.columns, "DataFrame must contain a 'low' column."
        assert 'close' in df.columns, "DataFrame must contain a 'close' column."
        assert 'volume' in df.columns, "DataFrame must contain a 'volume' column."

        windows = windows if windows is not None else self.windows
        level = level if level is not None else self.level

        factor_methods = {
            'kmid': self._kmid,
            'klen': self._klen,
            'kup': self._kup,
            'klow': self._klow,
            'ksft': self._ksft,
            'roc': self._roc,
            'ma': self._ma,
            'std': self._std,
            'beta': self._beta,
            'max': self._max,
            'min': self._min,
            'qtlu': self._qtlu,
            'qtld': self._qtld,
            'rank': self._rank,
            'imax': self._imax,
            'imin': self._imin,
            'imxd': self._imxd,
            'rsv': self._rsv,
            'cnt': self._cnt,
            'corr': self._corr,
            'cord': self._cord,
            'sum': self._sum,
            'vma': self._vma,
            'vstd': self._vstd,
            'wvma': self._wvma,
            'vsum': self._vsum,
            'logvol': self._logvol,
        }

        factors_info = {}
        result_df = pd.DataFrame(index=df.index)
        result_df['timestamp'] = df['timestamp']

        for factor_name, method in factor_methods.items():
            factor_df, factor_info = await method(df, windows=windows, level=level)
            logger.info(f"Computed factor: {list(factor_info.keys())} with windows: {windows} and level: {level}")
            result_df = pd.concat([result_df, factor_df], axis=1)
            factors_info.update(factor_info)

        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        result_df = result_df.fillna(0)

        assert sorted(factors_info.keys()) == self.names, \
            f"Factor names do not match. Expected: {self.names}, Got: {sorted(factors_info.keys())}"

        res = dict(
            factors_df=result_df,
            factors_info=factors_info,
        )

        return res