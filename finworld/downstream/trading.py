import numpy as np
from finworld.registry import DOWNSTREAM

@DOWNSTREAM.register_module(force=True)
class AlphaThresholdStrategy():
    def __init__(self,
                 alpha: float = 5.0,
                 **kwargs):
        self.alpha = alpha

    def get_actions(self, y_pred: np.ndarray):
        """
        Return the actions based on alpha thresholding.
        :param y_pred: Predicted rets, shape (N, )
        :return: actions （N, ), (0： sell, 1: hold, 2: buy)
        """
        actions = np.ones(y_pred.shape, dtype=int)

        low_quantile = np.percentile(y_pred, self.alpha)
        high_quantile = np.percentile(y_pred, 100 - self.alpha)

        actions[y_pred < low_quantile] = 0  # Sell
        actions[y_pred > high_quantile] = 2  # Buy

        return actions

    def __call__(self,
                 *args,
                 y_pred: np.ndarray,
                 timestamp: np.ndarray,
                 **kwargs):
        """
        Call method to execute the strategy.
        :param y_pred: Predicted labels, shape (B, S)
        """
        actions = self.get_actions(y_pred)
        res = {}
        for action, timestamp in zip(actions, timestamp):
            res[timestamp] = action
        return res

if __name__ == '__main__':
    strategy = AlphaThresholdStrategy(alpha=5.0)
    y_pred = np.array([0.01, 0.02, -0.01, -0.05, 0.03])
    timestamp = np.array(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    actions = strategy(y_pred=y_pred, timestamp=timestamp)
    print(actions)  # Expected output: array of actions based on alpha thresholding