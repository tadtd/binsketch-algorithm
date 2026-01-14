import numpy as np
from scipy.sparse import csr_matrix

def inner_product(sketch1: csr_matrix, sketch2: csr_matrix) -> float:
    """Compute Inner Product"""
    if sketch1.shape[1] != sketch2.shape[0]:
        raise ValueError("Incompatible shapes for inner product")
    
    return float((sketch1 @ sketch2).sum())

def hamming_distance(sketch1: csr_matrix, sketch2: csr_matrix) -> float:
    """Compute Hamming Distance"""
    if sketch1.shape != sketch2.shape:
        raise ValueError("Sketches must be of the same shape")
    
    return float((sketch1 != sketch2).sum())

def jaccard_similarity(sketch1: csr_matrix, sketch2: csr_matrix) -> float:
    """Compute Jaccard Similarity"""
    intersection = sketch1.minimum(sketch2).sum()
    union = sketch1.maximum(sketch2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / union

def cosine_similarity(sketch1: csr_matrix, sketch2: csr_matrix) -> float:
    """Compute Cosine Similarity"""
    dot_product = sketch1.multiply(sketch2).sum()
    norm1 = np.sqrt(sketch1.multiply(sketch1).sum())
    norm2 = np.sqrt(sketch2.multiply(sketch2).sum())
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product) / (norm1 * norm2)