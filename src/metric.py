import numpy as np

def mse(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the Mean Squared Error between two vectors."""
    if v1.shape != v2.shape:
        raise ValueError("Vectors must be of the same length")
    
    return float(np.mean((v1 - v2) ** 2))


def minus_log_mse(v1: np.ndarray, v2: np.ndarray) -> float:
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