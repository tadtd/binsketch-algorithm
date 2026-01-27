import numpy as np
from scipy.sparse import csr_matrix
from typing import Union
from .gpu_utils import get_array_module, to_cpu, GPU_AVAILABLE

# Try to import CuPy types for type checking
if GPU_AVAILABLE:
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    except ImportError:
        cp = None
        gpu_csr_matrix = None
else:
    cp = None
    gpu_csr_matrix = None

def _to_array(v: Union[np.ndarray, csr_matrix]) -> np.ndarray:
    """Convert input to numpy array if needed."""
    # Handle GPU arrays
    if GPU_AVAILABLE and cp is not None:
        if isinstance(v, cp.ndarray):
            return v
        if gpu_csr_matrix and isinstance(v, gpu_csr_matrix):
            return v.toarray()
    
    # Handle CPU arrays
    if isinstance(v, np.ndarray):
        return v
    return v.toarray()

def mse(v1: Union[np.ndarray, csr_matrix], v2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute the Mean Squared Error between two vectors."""
    a1 = _to_array(v1)
    a2 = _to_array(v2)
    
    if a1.shape != a2.shape:
        raise ValueError("Vectors must be of the same length")
    
    xp = get_array_module(a1)
    result = xp.mean((a1 - a2) ** 2)
    
    # Transfer to CPU if needed
    return float(to_cpu(result))


def minus_log_mse(v1: Union[np.ndarray, csr_matrix], v2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute the Negative Log Mean Squared Error between two vectors."""
    mse_value = mse(v1, v2)
    if mse_value == 0:
        return float('inf')  # Avoid log(0)
    
    return -np.log(mse_value)

def precision(true_positive: int, false_positive: int) -> float:
    """Compute Precision given true positives and false positives."""
    if true_positive + false_positive == 0:
        return 0.0
    
    return true_positive / (true_positive + false_positive)

def recall(true_positive: int, false_negative: int) -> float:
    """Compute Recall given true positives and false negatives."""
    if true_positive + false_negative == 0:
        return 0.0
    
    return true_positive / (true_positive + false_negative)

def f1_score(precision: float, recall: float) -> float:
    """Compute the F1 Score given precision and recall."""
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)