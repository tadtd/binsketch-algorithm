from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ..gpu_utils import (
    get_array_module, get_sparse_module, to_gpu, to_cpu,
    create_random_state, GPUConfig, concatenate, arange, ones
)

class BinSketch(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.P = None

    def _estimate_sparsity(self, a) -> float:
        """Estimates the original vector's sparsity from its sketch."""
        xp = get_array_module(a)
        n = a.shape[-1]
        spar_a = xp.count_nonzero(a)
        if spar_a / n < 1:
            spar_est_a = xp.round(xp.log(1-spar_a/n) / xp.log(1-1/n))
        else:
            spar_est_a = spar_a
        return float(spar_est_a)

    def mapping(self, X: csr_matrix, k: int, return_gpu: bool = False) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
            return_gpu: If True and GPU is enabled, return GPU array without transferring to CPU.
        """
        _, n = X.shape
        use_gpu = GPUConfig.is_enabled()
        
        # Transfer to GPU if enabled
        if use_gpu:
            X = to_gpu(X)
        
        # Cache projection matrix P if not exists or dimensions changed
        if not hasattr(self, 'P') or self.P is None or self.P.shape != (n, k):
            full_reps = n // k
            remainder = n % k
            
            rng = create_random_state(self.seed, use_gpu)
            xp = get_array_module()
            
            tile_part = xp.tile(xp.arange(k, dtype=xp.int32), full_reps)
            remainder_part = rng.choice(k, remainder, replace=False).astype(xp.int32)
            buckets = concatenate([tile_part, remainder_part], use_gpu=use_gpu)
            rng.shuffle(buckets)
            
            row_indices = arange(n, use_gpu=use_gpu, dtype=xp.int32)
            col_indices = buckets.astype(xp.int32)
            data = ones(n, dtype=xp.float32 if use_gpu else bool, use_gpu=use_gpu)
            
            sparse_module = get_sparse_module()
            self.P = sparse_module.csc_matrix((data, (row_indices, col_indices)), shape=(n, k))

        X_sketch = X.dot(self.P)
        
        # Convert to dense array and apply threshold (>0 -> 1)
        X_sketch_dense = X_sketch.toarray()
        xp = get_array_module(X_sketch_dense)
        
        if use_gpu:
            X_sketch_binary = xp.where(X_sketch_dense > 0, 1.0, 0.0).astype(xp.float32)
        else:
            X_sketch_binary = xp.where(X_sketch_dense > 0, 1, 0).astype(np.int8)
        
        if return_gpu and use_gpu:
            return X_sketch_binary
        
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
        
        xp = get_array_module(sketch1)
        
        n = sketch1.shape[-1]
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        
        # IP is simpler on dense arrays: element-wise mult and sum
        IP = xp.sum(sketch1 * sketch2)
        
        val = (1-1/n)**spar_est_1 + (1-1/n)**spar_est_2 - 1 + IP/n
        if val > 0:
            Bin_IP_est = xp.round(spar_est_1 + spar_est_2 - (xp.log(val)/xp.log(1-1/n)))
            if Bin_IP_est < 0:
                Bin_IP_est = int(to_cpu(IP))
            else:
                Bin_IP_est = float(to_cpu(Bin_IP_est))
        else:
            Bin_IP_est = int(to_cpu(IP))
        return float(Bin_IP_est)
    
    def estimate_hamming_distance(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Hamming distance.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError(f"Sketches must have the same shape: {sketch1.shape} vs {sketch2.shape}")
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        Bin_Ham_est = spar_est_1 + spar_est_2 - 2*est_ip
        return float(Bin_Ham_est)
    
    def estimate_jaccard_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Jaccard similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Jaccard similarity estimation.")
        Bin_IP_est = self.estimate_inner_product(sketch1, sketch2)
        Bin_Ham_est = self.estimate_hamming_distance(sketch1, sketch2)
        denom = Bin_Ham_est + Bin_IP_est
        if denom == 0:
            return 0.0
        Jaccard_est = Bin_IP_est / denom
        return float(Jaccard_est)
    
    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Cosine similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        xp = get_array_module(sketch1)
        
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        
        denom = float(to_cpu(xp.sqrt(spar_est_1) * xp.sqrt(spar_est_2)))
        if denom == 0:
            return 0.0
            
        Cosine_est = est_ip / denom
        return float(Cosine_est)
    
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
        n_pairs = len(pairs)
        n = sketches.shape[-1]
        
        # Extract indices for batch processing
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]  # shape: (n_pairs, sketch_dim)
        sketches_j = sketches[indices_j]  # shape: (n_pairs, sketch_dim)
        
        # Estimate sparsities in batch
        spar_i = xp.count_nonzero(sketches_i, axis=1).astype(xp.float32)
        spar_j = xp.count_nonzero(sketches_j, axis=1).astype(xp.float32)
        
        # Vectorized sparsity estimation
        mask_i = spar_i / n < 1
        mask_j = spar_j / n < 1
        
        spar_est_i = xp.where(mask_i, 
                              xp.round(xp.log(1 - spar_i/n) / xp.log(1 - 1/n)), 
                              spar_i)
        spar_est_j = xp.where(mask_j,
                              xp.round(xp.log(1 - spar_j/n) / xp.log(1 - 1/n)),
                              spar_j)
        
        # Batch inner product: element-wise multiply and sum along sketch dimension
        IP = xp.sum(sketches_i * sketches_j, axis=1)  # shape: (n_pairs,)
        
        # Vectorized estimation formula
        val = (1-1/n)**spar_est_i + (1-1/n)**spar_est_j - 1 + IP/n
        
        # Compute estimates
        Bin_IP_est = xp.where(val > 0,
                              xp.round(spar_est_i + spar_est_j - (xp.log(val)/xp.log(1-1/n))),
                              IP)
        
        # Fix negative estimates
        Bin_IP_est = xp.where(Bin_IP_est < 0, IP, Bin_IP_est)
        
        return to_cpu(Bin_IP_est).astype(np.float32)
    
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
        n = sketches.shape[-1]
        
        # Extract indices for batch processing
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Estimate sparsities in batch
        spar_i = xp.count_nonzero(sketches_i, axis=1).astype(xp.float32)
        spar_j = xp.count_nonzero(sketches_j, axis=1).astype(xp.float32)
        
        mask_i = spar_i / n < 1
        mask_j = spar_j / n < 1
        
        spar_est_i = xp.where(mask_i, 
                              xp.round(xp.log(1 - spar_i/n) / xp.log(1 - 1/n)), 
                              spar_i)
        spar_est_j = xp.where(mask_j,
                              xp.round(xp.log(1 - spar_j/n) / xp.log(1 - 1/n)),
                              spar_j)
        
        # Batch inner product estimation (reuse same logic)
        IP = xp.sum(sketches_i * sketches_j, axis=1)
        val = (1-1/n)**spar_est_i + (1-1/n)**spar_est_j - 1 + IP/n
        Bin_IP_est = xp.where(val > 0,
                              xp.round(spar_est_i + spar_est_j - (xp.log(val)/xp.log(1-1/n))),
                              IP)
        Bin_IP_est = xp.where(Bin_IP_est < 0, IP, Bin_IP_est)
        
        # Vectorized cosine calculation
        denom = xp.sqrt(spar_est_i) * xp.sqrt(spar_est_j)
        Cosine_est = xp.where(denom > 0, Bin_IP_est / denom, 0.0)
        
        return to_cpu(Cosine_est).astype(np.float32)
    
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
        n = sketches.shape[-1]
        
        # Extract indices for batch processing
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Estimate sparsities in batch
        spar_i = xp.count_nonzero(sketches_i, axis=1).astype(xp.float32)
        spar_j = xp.count_nonzero(sketches_j, axis=1).astype(xp.float32)
        
        mask_i = spar_i / n < 1
        mask_j = spar_j / n < 1
        
        spar_est_i = xp.where(mask_i, 
                              xp.round(xp.log(1 - spar_i/n) / xp.log(1 - 1/n)), 
                              spar_i)
        spar_est_j = xp.where(mask_j,
                              xp.round(xp.log(1 - spar_j/n) / xp.log(1 - 1/n)),
                              spar_j)
        
        # Batch inner product estimation
        IP = xp.sum(sketches_i * sketches_j, axis=1)
        val = (1-1/n)**spar_est_i + (1-1/n)**spar_est_j - 1 + IP/n
        Bin_IP_est = xp.where(val > 0,
                              xp.round(spar_est_i + spar_est_j - (xp.log(val)/xp.log(1-1/n))),
                              IP)
        Bin_IP_est = xp.where(Bin_IP_est < 0, IP, Bin_IP_est)
        
        # Hamming distance estimation
        Bin_Ham_est = spar_est_i + spar_est_j - 2 * Bin_IP_est
        
        # Jaccard calculation
        denom = Bin_Ham_est + Bin_IP_est
        Jaccard_est = xp.where(denom > 0, Bin_IP_est / denom, 0.0)
        
        return to_cpu(Jaccard_est).astype(np.float32)
    