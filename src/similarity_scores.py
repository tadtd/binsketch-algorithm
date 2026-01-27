import numpy as np
from scipy.sparse import csr_matrix
from typing import Union
from .gpu_utils import get_array_module, get_sparse_module, to_cpu, GPU_AVAILABLE

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

def _to_sparse(x):
    """Convert input to sparse matrix if needed, handling both CPU and GPU arrays."""
    # Check if already sparse
    if isinstance(x, csr_matrix):
        return x
    
    # Check if GPU sparse matrix
    if gpu_csr_matrix and isinstance(x, gpu_csr_matrix):
        return x
    
    # Convert dense to sparse using appropriate module
    xp = get_array_module(x)
    sparse_module = get_sparse_module(x)
    return sparse_module.csr_matrix(x)

def inner_product(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Inner Product (supports both CPU and GPU arrays)"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    if s1.shape != s2.shape:
        raise ValueError("Incompatible shapes for inner product")
    
    result = (s1.multiply(s2)).sum()
    # Ensure we return a Python float
    return float(to_cpu(result) if hasattr(result, 'get') else result)

def hamming_distance(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Hamming Distance (supports both CPU and GPU arrays)"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    if s1.shape != s2.shape:
        raise ValueError("Sketches must be of the same shape")
    
    result = (s1 != s2).sum()
    return float(to_cpu(result) if hasattr(result, 'get') else result)

def jaccard_similarity(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Jaccard Similarity (supports both CPU and GPU arrays)"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    intersection = s1.minimum(s2).sum()
    union = s1.maximum(s2).sum()
    
    # Transfer to CPU if needed
    intersection = to_cpu(intersection) if hasattr(intersection, 'get') else intersection
    union = to_cpu(union) if hasattr(union, 'get') else union
    
    if union == 0:
        return 0.0
    
    return float(intersection) / union

def cosine_similarity(sketch1: Union[np.ndarray, csr_matrix], sketch2: Union[np.ndarray, csr_matrix]) -> float:
    """Compute Cosine Similarity (supports both CPU and GPU arrays)"""
    s1 = _to_sparse(sketch1)
    s2 = _to_sparse(sketch2)
    dot_product = s1.multiply(s2).sum()
    
    # Get the appropriate array module
    xp = get_array_module(s1.data if hasattr(s1, 'data') else None)
    
    norm1 = xp.sqrt(s1.multiply(s1).sum())
    norm2 = xp.sqrt(s2.multiply(s2).sum())
    
    # Transfer to CPU if needed
    dot_product = to_cpu(dot_product) if hasattr(dot_product, 'get') else dot_product
    norm1 = to_cpu(norm1) if hasattr(norm1, 'get') else norm1
    norm2 = to_cpu(norm2) if hasattr(norm2, 'get') else norm2
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product) / (norm1 * norm2)