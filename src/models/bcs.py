from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from ..gpu_utils import (
    get_array_module, get_sparse_module, to_gpu, to_cpu,
    create_random_state, GPUConfig, arange, ones
)

class BinaryCompressionSchema(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.P = None

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n_features = X.shape
        use_gpu = GPUConfig.is_enabled()
        
        # Transfer to GPU if enabled
        if use_gpu:
            X = to_gpu(X)
        
        # Cache projection matrix P if not exists or dimensions changed
        if not hasattr(self, 'P') or self.P is None or self.P.shape != (n_features, k):
            rng = create_random_state(self.seed, use_gpu)
            xp = get_array_module()
            
            buckets = rng.randint(0, k, size=n_features).astype(xp.int32)
            row_indices = arange(n_features, use_gpu=use_gpu, dtype=xp.int32)
            col_indices = buckets.astype(xp.int32)
            data = ones(n_features, dtype=xp.float32 if use_gpu else int, use_gpu=use_gpu)
            
            sparse_module = get_sparse_module()
            self.P = sparse_module.coo_matrix((data, (row_indices, col_indices)), shape=(n_features, k))

        X_sketch = X.dot(self.P)
        
        X_sketch_dense = X_sketch.toarray()
        xp = get_array_module(X_sketch_dense)
        
        if use_gpu:
            X_sketch_binary = (X_sketch_dense.astype(xp.float32) % 2)
        else:
            X_sketch_binary = X_sketch_dense.astype(int) % 2
        
        result = to_cpu(X_sketch_binary)
        if use_gpu:
            result = result.astype(np.int8)
        return result

    def estimate_inner_product(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates inner product.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Inner Product estimation.")
        
        # For binary vectors, inner product is sum of AND
        return float(np.sum(sketch1 * sketch2))

    def estimate_hamming_distance(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Hamming distance.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Hamming distance estimation.")
        
        # XOR sum
        diff = np.bitwise_xor(sketch1, sketch2)
        return float(np.count_nonzero(diff))

    def estimate_jaccard_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Jaccard similarity: Intersection / Union
        Union = Hamming + Intersection
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Jaccard similarity estimation.")
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        est_hamming = self.estimate_hamming_distance(sketch1, sketch2)
        
        union = est_hamming + est_ip
        
        if union == 0:
            return 0.0
            
        return est_ip / union

    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Cosine similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        est_sq_norm1 = self.estimate_inner_product(sketch1, sketch1)
        est_sq_norm2 = self.estimate_inner_product(sketch2, sketch2)
        
        denom = np.sqrt(est_sq_norm1) * np.sqrt(est_sq_norm2)
        
        if denom == 0:
            return 0.0
            
        return est_ip / denom