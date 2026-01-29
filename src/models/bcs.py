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

    def mapping(self, X: csr_matrix, k: int, return_gpu: bool = False) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
            return_gpu: If True and GPU is enabled, return GPU array without transferring to CPU.
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
        
        # Return GPU array if requested, otherwise transfer to CPU
        if return_gpu and use_gpu:
            return X_sketch_binary
        
        result = to_cpu(X_sketch_binary)
        if use_gpu:
            result = result.astype(np.int8)
        return result

    def estimate_inner_product(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates inner product.
        Supports both CPU and GPU arrays.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Inner Product estimation.")
        
        from ..gpu_utils import get_array_module, to_cpu
        xp = get_array_module(sketch1)
        
        # For binary vectors, inner product is sum of AND
        result = xp.sum(sketch1 * sketch2)
        return float(to_cpu(result))

    def estimate_hamming_distance(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Hamming distance.
        Supports both CPU and GPU arrays.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Hamming distance estimation.")
        
        from ..gpu_utils import get_array_module, to_cpu
        xp = get_array_module(sketch1)
        
        # XOR sum
        diff = xp.bitwise_xor(sketch1, sketch2)
        result = xp.count_nonzero(diff)
        return float(to_cpu(result))

    def estimate_jaccard_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Jaccard similarity: Intersection / Union
        Jaccard = |A ∩ B| / |A ∪ B| = IP / (|A| + |B| - IP)
        Supports both CPU and GPU arrays.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Jaccard similarity estimation.")
        
        from ..gpu_utils import get_array_module, to_cpu
        xp = get_array_module(sketch1)
        
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        cardinality1 = to_cpu(xp.count_nonzero(sketch1))
        cardinality2 = to_cpu(xp.count_nonzero(sketch2))
        union = cardinality1 + cardinality2 - est_ip
        
        if union == 0:
            return 0.0
            
        return est_ip / union

    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Cosine similarity.
        Supports both CPU and GPU arrays.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        
        from ..gpu_utils import get_array_module, to_cpu
        xp = get_array_module(sketch1)
        
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        est_sq_norm1 = self.estimate_inner_product(sketch1, sketch1)
        est_sq_norm2 = self.estimate_inner_product(sketch2, sketch2)
        
        denom = xp.sqrt(est_sq_norm1) * xp.sqrt(est_sq_norm2)
        
        if denom == 0:
            return 0.0
            
        result = est_ip / denom
        return float(to_cpu(result))
    
    def estimate_inner_product_batch(self, sketches: np.ndarray, pairs: list) -> np.ndarray:
        """
        Estimates inner products for multiple pairs efficiently using GPU vectorization.
        
        Args:
            sketches: Array of sketches (n_samples, sketch_dim) - can be on GPU or CPU
            pairs: List of (i, j) tuples indicating which pairs to estimate
            
        Returns:
            Array of estimated inner products for each pair
        """
        xp = get_array_module(sketches)
        
        # Extract indices for batch processing
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Batch inner product: element-wise multiply and sum
        result = xp.sum(sketches_i * sketches_j, axis=1)
        
        return to_cpu(result).astype(np.float32)
    
    def estimate_cosine_similarity_batch(self, sketches: np.ndarray, pairs: list) -> np.ndarray:
        """
        Estimates cosine similarities for multiple pairs efficiently using GPU vectorization.
        
        Args:
            sketches: Array of sketches (n_samples, sketch_dim) - can be on GPU or CPU
            pairs: List of (i, j) tuples indicating which pairs to estimate
            
        Returns:
            Array of estimated cosine similarities for each pair
        """
        xp = get_array_module(sketches)
        
        # Extract indices
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Batch inner products
        est_ip = xp.sum(sketches_i * sketches_j, axis=1)
        est_sq_norm1 = xp.sum(sketches_i * sketches_i, axis=1)
        est_sq_norm2 = xp.sum(sketches_j * sketches_j, axis=1)
        
        # Vectorized cosine calculation
        denom = xp.sqrt(est_sq_norm1) * xp.sqrt(est_sq_norm2)
        cosine_est = xp.where(denom > 0, est_ip / denom, 0.0)
        
        return to_cpu(cosine_est).astype(np.float32)
    
    def estimate_jaccard_similarity_batch(self, sketches: np.ndarray, pairs: list) -> np.ndarray:
        """
        Estimates Jaccard similarities for multiple pairs efficiently using GPU vectorization.
        
        Args:
            sketches: Array of sketches (n_samples, sketch_dim) - can be on GPU or CPU
            pairs: List of (i, j) tuples indicating which pairs to estimate
            
        Returns:
            Array of estimated Jaccard similarities for each pair
        """
        xp = get_array_module(sketches)
        
        # Extract indices
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Batch inner products and cardinalities
        est_ip = xp.sum(sketches_i * sketches_j, axis=1)
        cardinality1 = xp.count_nonzero(sketches_i, axis=1)
        cardinality2 = xp.count_nonzero(sketches_j, axis=1)
        
        # Vectorized Jaccard calculation
        union = cardinality1 + cardinality2 - est_ip
        jaccard_est = xp.where(union > 0, est_ip / union, 0.0)
        
        return to_cpu(jaccard_est).astype(np.float32)
        return float(to_cpu(result))