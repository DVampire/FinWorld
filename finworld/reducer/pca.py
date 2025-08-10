from typing import Dict, Any
from sklearn.decomposition import PCA as SKPCA

from finworld.registry import REDUCER

@REDUCER.register_module(force=True)
class PCA:
    def __init__(self,
                 params: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the PCA reducer with optional parameters.

        Args:
            params (Dict[str, Any], optional): Parameters for PCA. Defaults to None.
        """
        self.params = params if params is not None else {}
        self.pca = SKPCA(**self.params)

    def fit(self, data: Any) -> None:
        """
        Fit the PCA model to the provided data.

        Args:
            data (Any): Data to fit the PCA model.
        """
        self.pca.fit(data)

    def fit_transform(self, data: Any) -> Any:
        """
        Fit the PCA model and transform the data.

        Args:
            data (Any): Data to fit and transform.

        Returns:
            Any: Transformed data.
        """
        self.fit(data)
        return self.pca.transform(data)

    def transform(self, data: Any) -> Any:
        """
        Transform the data using the fitted PCA model.

        Args:
            data (Any): Data to transform.

        Returns:
            Any: Transformed data.
        """
        return self.pca.transform(data)