import json
import os
import numpy as np

_BIAS_PATH = os.path.join(os.path.dirname(__file__), 'bias.json')

_global_bias = np.zeros(99)
if os.path.exists(_BIAS_PATH):
    try:
        _global_bias = np.load(_BIAS_PATH)
    except Exception:
        pass

def set_global_bias(vec: np.ndarray) -> None:
    global _global_bias
    _global_bias = np.asarray(vec, dtype=float)
    np.save(_BIAS_PATH, _global_bias)

def get_global_bias() -> np.ndarray:
    return _global_bias
