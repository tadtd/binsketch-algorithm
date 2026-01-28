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


# ============================================================================
# VECTORIZED BATCH OPERATIONS (for GPU acceleration)
# ============================================================================

def compute_similarity_matrix(X1: np.ndarray, X2: np.ndarray, similarity_type: str) -> np.ndarray:
    """
    Compute pairwise similarity matrix between two sets of vectors using GPU acceleration.
    
    Args:
        X1: First set of vectors (n1, d)
        X2: Second set of vectors (n2, d)
        similarity_type: 'inner_product', 'cosine', or 'jaccard'
        
    Returns:
        Similarity matrix (n1, n2)
    """
    xp = get_array_module(X1)
    
    if similarity_type == 'inner_product':
        return xp.dot(X1, X2.T)
    
    elif similarity_type == 'cosine':
        # Normalize vectors
        norms1 = xp.sqrt(xp.sum(X1 ** 2, axis=1, keepdims=True))
        norms2 = xp.sqrt(xp.sum(X2 ** 2, axis=1, keepdims=True))
        norms1 = xp.maximum(norms1, 1e-10)
        norms2 = xp.maximum(norms2, 1e-10)
        X1_norm = X1 / norms1
        X2_norm = X2 / norms2
        return xp.dot(X1_norm, X2_norm.T)
    
    elif similarity_type == 'jaccard':
        # For binary vectors: intersection / union
        intersection = xp.dot(X1, X2.T)
        cardinalities1 = xp.sum(X1, axis=1, keepdims=True)
        cardinalities2 = xp.sum(X2, axis=1, keepdims=True)
        union = cardinalities1 + cardinalities2.T - intersection
        union = xp.maximum(union, 1e-10)
        return intersection / union
    
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")


def batch_inner_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute inner product between all pairs of rows in X and Y.
    GPU-accelerated using matrix multiplication.
    
    Args:
        X: Matrix (n, d)
        Y: Matrix (m, d)
        
    Returns:
        Inner product matrix (n, m)
    """
    xp = get_array_module(X)
    return xp.dot(X, Y.T)


def batch_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all pairs of rows in X and Y.
    GPU-accelerated using vectorized operations.
    
    Args:
        X: Matrix (n, d)
        Y: Matrix (m, d)
        
    Returns:
        Cosine similarity matrix (n, m)
    """
    xp = get_array_module(X)
    
    # Compute norms
    norms_X = xp.sqrt(xp.sum(X ** 2, axis=1, keepdims=True))
    norms_Y = xp.sqrt(xp.sum(Y ** 2, axis=1, keepdims=True))
    
    # Avoid division by zero
    norms_X = xp.maximum(norms_X, 1e-10)
    norms_Y = xp.maximum(norms_Y, 1e-10)
    
    # Normalize
    X_normalized = X / norms_X
    Y_normalized = Y / norms_Y
    
    # Compute dot product
    return xp.dot(X_normalized, Y_normalized.T)


def batch_jaccard_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard similarity between all pairs of rows in X and Y.
    GPU-accelerated for binary vectors using vectorized operations.
    
    Args:
        X: Binary matrix (n, d)
        Y: Binary matrix (m, d)
        
    Returns:
        Jaccard similarity matrix (n, m)
    """
    xp = get_array_module(X)
    
    # Intersection = X @ Y.T (for binary vectors, this is the dot product)
    intersection = xp.dot(X, Y.T)
    
    # Union = |X| + |Y| - intersection
    cardinalities_X = xp.sum(X, axis=1, keepdims=True)
    cardinalities_Y = xp.sum(Y, axis=1, keepdims=True)
    union = cardinalities_X + cardinalities_Y.T - intersection
    
    # Avoid division by zero
    union = xp.maximum(union, 1e-10)
    
    return intersection / union


# ============================================================================
# PAIRWISE OPERATIONS (for individual vector pairs)
# ============================================================================

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