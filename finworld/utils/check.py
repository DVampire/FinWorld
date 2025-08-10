import numpy as np
from typing import List, Tuple, Any
import torch

def check_data(value: Any):
    if isinstance(value, torch.Tensor):
        return f"Type: torch.Tensor, Dtype: {value.dtype}, Shape: {value.shape}"
    elif isinstance(value, np.ndarray):
        return f"Type: np.ndarray, Dtype: {value.dtype}, Shape: {value.shape}"
    elif isinstance(value, (int, float, str)):
        return f"Type: {type(value)}, Value: {value}"
    elif isinstance(value, (List, Tuple)):
        return f"Type: {type(value)}, Length: {len(value)}"
    else:
        return f"Type: {type(value)}"