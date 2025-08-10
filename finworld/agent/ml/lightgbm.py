import numpy as np
from typing import List, Union, Dict, Any
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from finworld.registry import AGENT
from finworld.registry import REDUCER
from finworld.task import TaskType
from finworld.utils import TimeLevel, TimeLevelFormat
from finworld.log import logger

@AGENT.register_module(force=True)
class Lightgbm():
    def __init__(self,
                 task_type: str,
                 level: str = "1day",
                 num_regressors: int = 1,
                 reducer: Dict = None,
                 params: Dict = None,
                 **kwargs
                 ):
        super(Lightgbm, self).__init__()

        self.task_type = TaskType.from_string(task_type)
        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)
        self.num_regressors = num_regressors

        self.params = params if params is not None else {}

        if self.task_type == TaskType.TRADING:
            self.model = LGBMRegressor(
                **self.params,
            )
            self.reducer = REDUCER.build(reducer) if reducer is not None else None
        elif self.task_type == TaskType.PORTFOLIO:
            self.models = [LGBMRegressor(
                **self.params,
            ) for _ in range(num_regressors)]
            self.reducers = [REDUCER.build(reducer) if reducer is not None else None
                             for _ in range(num_regressors)]


    def __str__(self):
        return f"Lightgbm(task_type={self.task_type.value})"

    def __repr__(self):
        return self.__str__()

    def train(self,
              data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
              valid_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]] = None,
              ) -> None:
        """
        Train the LightGBM model on the provided data.

        Args:
            data (pd.DataFrame): The input data containing stock prices and other features.
                The DataFrame should have a column for stock prices, typically named "price".
        """

        x = data["features"]
        y = data["labels"]

        if self.task_type == TaskType.PORTFOLIO:
            logger.info(f"Training {self.num_regressors} models for portfolio...")

            for index, (key, x_) in enumerate(x.items()):
                logger.info(f"Training model for {key}...")

                model = self.models[index]
                reducer = self.reducers[index]
                y_ = y[key]

                if reducer is not None:
                    x_ = reducer.fit_transform(x_)

                if valid_data is not None:
                    valid_x_ = valid_data["features"][key]
                    valid_y_ = valid_data["labels"][key]

                    if reducer is not None:
                        valid_x_ = reducer.transform(valid_x_)

                    model.fit(
                        x_, y_,
                        eval_set=[(valid_x_, valid_y_)],
                        eval_metric="l2",
                        callbacks=[
                            lgb.log_evaluation(100)
                        ]
                    )
                else:
                    model.fit(x_, y_)

        elif self.task_type == TaskType.TRADING:
            logger.info("Training LightGBM model for trading...")

            if self.reducer is not None:
                x = self.reducer.fit_transform(x)

            if valid_data is not None:
                valid_x = valid_data["features"].values
                valid_y = valid_data["labels"].values

                if self.reducer is not None:
                    valid_x = self.reducer.transform(valid_x)

                self.model.fit(
                    x, y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric="l2",
                    callbacks=[
                        lgb.log_evaluation(100)
                    ]
                )
            else:
                self.model.fit(x, y)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def forward(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Any:
        """
        Buy and hold strategy: always buy the stock at the first time step and hold it.

        Args:
            data (pd.DataFrame): The input data containing stock prices and other features.
                The DataFrame should have a column for stock prices, typically named "price".

        Returns:
            Dict: A dictionary with timestamps as keys and actions as values.
        """
        pred = None
        if self.task_type == TaskType.PORTFOLIO:
            logger.info("Predicting with LightGBM for portfolio...")

            features = data["features"]
            pred = []
            for index, (key, features_) in enumerate(features.items()):
                logger.info(f"Predicting for {key}...")

                model = self.models[index]
                reducer = self.reducers[index]

                if reducer is not None:
                    features_ = reducer.transform(features_)

                pred_ = model.predict(features_)
                pred.append(pred_)
            pred = np.stack(pred, axis=-1)

        elif self.task_type == TaskType.TRADING:
            logger.info("Predicting with LightGBM for trading...")

            features = data["features"]
            if self.reducer is not None:
                features = self.reducer.transform(features)
            pred = self.model.predict(features)
        return pred

