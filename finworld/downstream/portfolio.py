import numpy as np
from finworld.registry import DOWNSTREAM

@DOWNSTREAM.register_module(force=True)
class TopkDropoutStrategy():
    def __init__(self,
                 topk: int = 5,
                 dropout: int = 3,
                 **kwargs):
        self.topk = topk
        self.dropout = dropout

    def get_actions(self, y_pred: np.ndarray):
        """
        Return the holding assets' weights (N, S), each row is a portfolio weight vector.
        :param y_pred: Predicted labels, shape (N, S)
        :return: actions, np.ndarray, shape (N, S), where each row sums to 1 or is equally allocated to topk assets
        """
        N, S = y_pred.shape
        actions = np.zeros((N, S))
        topk_list = np.argsort(y_pred[0])[-self.topk:]
        actions[0, topk_list] = 1.0 / self.topk

        for i in range(1, N):
            pred_label = y_pred[i].flatten()
            pre_topk_list = topk_list

            same_assets_set = set(pre_topk_list) & set(np.argsort(pred_label)[-self.topk:])
            hold_assets = list(same_assets_set)

            if len(hold_assets) < (self.topk - self.dropout):
                remaining_assets = list(set(pre_topk_list) - same_assets_set)
                sorted_remaining = sorted(remaining_assets, key=lambda x: pred_label[x])
                hold_assets.extend(sorted_remaining[-(self.topk - self.dropout - len(hold_assets)):])

            non_hold_assets = list(set(range(S)) - set(hold_assets))
            sorted_non_hold = sorted(non_hold_assets, key=lambda x: pred_label[x])
            topk_candidates = hold_assets + sorted_non_hold[-(self.topk - len(hold_assets)):]
            topk_list = sorted(topk_candidates, key=lambda x: pred_label[x])[-self.topk:]

            actions[i, topk_list] = 1.0 / self.topk

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

        # Set cash positions
        cashes = np.zeros((actions.shape[0], 1))
        actions = np.concatenate([cashes, actions], axis=1)

        res = {}
        for action, timestamp in zip(actions, timestamp):
            res[timestamp] = action
        return res


if __name__ == '__main__':
    y_pred = np.random.rand(4, 5) * 0.01
    timestamp = np.array(['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04',
                         '2023-10-05'])

    strategy = TopkDropoutStrategy()
    actions = strategy(y_pred=y_pred, timestamp=timestamp)
    print(actions)