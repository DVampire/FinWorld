import os
import joblib
from typing import Any

def save_joblib(obj: Any, path: str):
    assert obj is not None, 'Object to save should not be None'
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)

def load_joblib(path: str) -> Any:
    if path and os.path.exists(path):
        return joblib.load(path)
    else:
        return None