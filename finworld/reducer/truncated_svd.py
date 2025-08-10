from typing import Dict, Any
from sklearn.decomposition import TruncatedSVD as SKSVD

from finworld.registry import REDUCER

@REDUCER.register_module(force=True)
class TruncatedSVD:
    def __init__(self,
                 params: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the TruncatedSVD reducer with optional parameters.

        Args:
            params (Dict[str, Any], optional): Parameters for svd. Defaults to None.
        """
        self.params = params if params is not None else {}
        self.svd = SKSVD(**self.params)

    def fit(self, data: Any) -> None:
        """
        Fit the TruncatedSVD model to the provided data.

        Args:
            data (Any): Data to fit the svd model.
        """
        self.svd.fit(data)

    def fit_transform(self, data: Any) -> Any:
        """
        Fit the TruncatedSVD model and transform the data.

        Args:
            data (Any): Data to fit and transform.

        Returns:
            Any: Transformed data.
        """
        self.fit(data)
        return self.svd.transform(data)

    def transform(self, data: Any) -> Any:
        """
        Transform the data using the fitted TruncatedSVD model.

        Args:
            data (Any): Data to transform.

        Returns:
            Any: Transformed data.
        """
        return self.svd.transform(data)