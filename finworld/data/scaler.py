import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from finworld.registry import SCALER
from finworld.utils import assemble_project_path

EPS = 1e-12


class BaseScaler:
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.mean = mean
        self.std = std

    def _prepared_mean_std(self,
                           start_index: Optional[int] = None,
                           end_index: Optional[int] = None,
                           ) -> Tuple[np.ndarray, np.ndarray]:
        if (start_index is not None and end_index is not None) and (
            len(self.mean.shape) != 1 and len(self.std.shape) != 1
        ):
            mean = self.mean[start_index : end_index + 1, ...]
            std = self.std[start_index : end_index + 1, ...]
        else:
            mean = self.mean
            std = self.std

        return mean, std

    def _convert(self, df: DataFrame | np.ndarray) -> Tuple[np.ndarray, List[str]]:
        df = deepcopy(df)

        columns = []
        if isinstance(df, DataFrame):
            values = df.values
            columns = list(df.columns)
        elif isinstance(df, np.ndarray):
            values = df
        else:
            raise ValueError('df should be in DataFrame or np.ndarray')
        return values, columns

    def _set_values(
        self, df: DataFrame | np.ndarray, values: np.ndarray, columns: List[str]
    ) -> DataFrame | np.ndarray:
        df = deepcopy(df)
        if isinstance(df, DataFrame):
            df[columns] = values
        elif isinstance(df, np.ndarray):
            df = values
        return df

    def fit_transform(self, df: DataFrame | np.ndarray) -> DataFrame | np.ndarray:
        raise NotImplementedError

    def transform(
            self,
            df: DataFrame | np.ndarray,
            start_index: Optional[int] = None,
            end_index: Optional[int] = None,
        ) -> DataFrame | np.ndarray:

        assert (
            self.mean is not None and self.std is not None
        ), 'mean and std should not be None'

        values, columns = self._convert(df)

        mean, std = self._prepared_mean_std(start_index, end_index)
        normed_values = 1.0 * (values - mean) / (std + EPS)

        df = self._set_values(df, normed_values, columns)

        return df

    def inverse_transform(
                self,
                df: DataFrame | np.ndarray,
                start_index: Optional[int] = None,
                end_index: Optional[int] = None,
            ) -> DataFrame | np.ndarray:
        assert (
            self.mean is not None and self.std is not None
        ), 'mean and std should not be None'

        normed_values, columns = self._convert(df)

        mean, std = self._prepared_mean_std(start_index, end_index)
        values = 1.0 * normed_values * std + mean

        df = self._set_values(df, values, columns)

        return df


@SCALER.register_module(force=True)
class StandardScaler(BaseScaler):
    def __init__(
        self,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        super(StandardScaler, self).__init__(mean, std)
        self.mean = mean
        self.std = std

    def fit_transform(self, df: DataFrame | np.ndarray) -> DataFrame | np.ndarray:
        values, columns = self._convert(df)

        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)

        mean = mean.repeat(repeats=len(values), axis=0)
        std = std.repeat(repeats=len(values), axis=0)

        normed_values = 1.0 * (values - mean) / (std + EPS)

        df = self._set_values(df, normed_values, columns)

        self.mean = mean
        self.std = std

        return df


@SCALER.register_module(force=True)
class WindowedScaler(BaseScaler):
    def __init__(
        self,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        window_size: int = 64,
    ):
        super(WindowedScaler, self).__init__(mean, std)
        self.mean = mean
        self.std = std
        self.window_size = window_size

    def fit_transform(self, df: DataFrame | np.ndarray) -> DataFrame | np.ndarray:
        values, columns = self._convert(df)

        nums, _ = values.shape[0], values.shape[1]
        block_nums = int(np.ceil(nums / self.window_size))

        def cal_mean_std(index, window_size=self.window_size):
            chunk = values[
                index * window_size : min((index + 1) * window_size, len(values))
            ]
            chunk_mean = chunk.mean(axis=0)
            chunk_std = chunk.std(axis=0)
            return chunk_mean, chunk_std

        with ThreadPoolExecutor(max_workers=min(32, 8)) as executor:
            mean_std = list(executor.map(cal_mean_std, range(block_nums)))

        mean = np.array([item[0] for item in mean_std])
        std = np.array([item[1] for item in mean_std])

        mean = np.concatenate([np.array(mean[0]).reshape(1, -1), mean[:-1]], axis=0)
        std = np.concatenate([np.array(std[0]).reshape(1, -1), std[:-1]], axis=0)

        # repeat mean and std
        mean = mean.repeat(repeats=self.window_size, axis=0)
        std = std.repeat(repeats=self.window_size, axis=0)
        mean = mean[:nums, ...]
        std = std[:nums, ...]

        normed_values = 1.0 * (values - mean) / (std + EPS)

        df = self._set_values(df, normed_values, columns)

        self.mean = mean
        self.std = std

        return df


__all__ = ['StandardScaler', 'WindowedScaler']

if __name__ == '__main__':
    df = pd.read_csv(
        assemble_project_path('datasets/processd_day_dj30/features/AAPL.csv')
    )
    df = df[['open', 'high', 'low', 'close']]
    print(df)

    std_scaler = StandardScaler()
    df = std_scaler.fit_transform(df)
    print(df)
    print(std_scaler.mean.shape, std_scaler.std.shape)

    df = std_scaler.inverse_transform(df, start_index=0, end_index=1)
    print(df)

    joblib.dump(std_scaler, 'tmp.joblib')
    scaler = joblib.load('tmp.joblib')
    os.remove('tmp.joblib')
    print(scaler)

    wd_scaler = WindowedScaler()
    df = wd_scaler.fit_transform(df)
    print(df)
    print(wd_scaler.mean.shape, wd_scaler.std.shape)

    df = wd_scaler.inverse_transform(df)
    print(df)

    joblib.dump(wd_scaler, 'tmp.joblib')
    wd_scaler = joblib.load('tmp.joblib')
    os.remove('tmp.joblib')
    print(wd_scaler)
