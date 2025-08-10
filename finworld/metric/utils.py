import torch
from typing import Union
import numpy as np

def clean_invalid_values(arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:

    """Clean invalid values (NaN and infinite) from a NumPy array or PyTorch tensor.
    Args:
        arr (Union[np.ndarray, torch.Tensor]): Input array or tensor.
    Returns:
        Union[np.ndarray, torch.Tensor]: Processed array or tensor with NaN and infinite values removed.
    """

    if isinstance(arr, torch.Tensor):
        mask = torch.isfinite(arr)
    elif isinstance(arr, np.ndarray):
        mask = np.isfinite(arr)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

    arr = arr[mask]

    return arr

def fill_invalid_values(arr: Union[np.ndarray, torch.Tensor], fill_value: float = 0.0) -> Union[np.ndarray, torch.Tensor]:
    """
    Fill invalid values (NaN and infinite) in a NumPy array or PyTorch tensor with a specified fill value.
    Args:
        arr:
        fill_value:

    Returns:
        Union[np.ndarray, torch.Tensor]: Processed array or tensor with invalid values replaced by fill_value.
    """

    if isinstance(arr, torch.Tensor):
        arr = torch.nan_to_num(arr, nan=.0, posinf=1.0, neginf=.0)
    elif isinstance(arr, np.ndarray):
        arr = np.nan_to_num(arr, nan=.0, posinf=1.0, neginf=.0)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    return arr