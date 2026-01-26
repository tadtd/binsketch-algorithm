import numpy as np
from scipy.sparse import csr_matrix
from typing import Union

def _to_sparse(x: Union[np.ndarray, csr_matrix]) -> csr_matrix:
    """Convert input to sparse matrix if needed."""
    if isinstance(x, csr_matrix):
        return x
    return csr_matrix(x)

def inner_product(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Inner Product"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    if s1.shape[1] != s2.shape[0]:
        raise ValueError("Incompatible shapes for inner product")
    
    return float((s1 @ s2).sum())

def hamming_distance(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Hamming Distance"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    if s1.shape != s2.shape:
        raise ValueError("Sketches must be of the same shape")
    
    return float((s1 != s2).sum())

def jaccard_similarity(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Jaccard Similarity"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    intersection = s1.minimum(s2).sum()
    union = s1.maximum(s2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / union

def cosine_similarity(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Cosine Similarity"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    dot_product = s1.multiply(s2).sum()
    norm1 = np.sqrt(s1.multiply(s1).sum())
    norm2 = np.sqrt(s2.multiply(s2).sum())
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product) / (norm1 * norm2)