from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class BinSketch(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.P = None

    def _estimate_sparsity(self, a: np.ndarray) -> float:
        """Estimates the original vector's sparsity from its sketch."""
        n = a.shape[-1]
        spar_a = np.count_nonzero(a)
        if spar_a / n < 1:
            spar_est_a = np.round(np.log(1-spar_a/n) / np.log(1-1/n))
        else:
            spar_est_a = spar_a
        return spar_est_a

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n = X.shape
        
        # Cache projection matrix P if not exists or dimensions changed
        if not hasattr(self, 'P') or self.P is None or self.P.shape != (n, k):
            full_reps = n // k
            remainder = n % k
            buckets = np.concatenate([
                np.tile(np.arange(k), full_reps),       # Repeated pattern
                np.random.RandomState(seed=self.seed).choice(k, remainder, replace=False) # Random remainder
            ])
            rng = np.random.RandomState(seed=self.seed)
            rng.shuffle(buckets)
            row_indices = np.arange(n) # Original columns 0..n-1
            col_indices = buckets      # Assigned buckets
            data = np.ones(n, dtype=bool) # All ones
            
            self.P = csc_matrix((data, (row_indices, col_indices)), shape=(n, k))

        X_sketch = X.dot(self.P)
        # Convert to dense array and apply threshold (>0 -> 1)
        X_sketch_dense = X_sketch.toarray()
        X_sketch_binary = np.where(X_sketch_dense > 0, 1, 0).astype(np.int8)
        return X_sketch_binary
    
    def estimate_inner_product(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates inner product.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Inner Product estimation.")
        n = sketch1.shape[-1]
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        
        # IP is simpler on dense arrays: element-wise mult and sum
        IP = int(np.sum(sketch1 * sketch2))
        
        val = (1-1/n)**spar_est_1 + (1-1/n)**spar_est_2 - 1 + IP/n
        if val > 0:
            Bin_IP_est = np.round(spar_est_1 + spar_est_2 - (np.log(val)/np.log(1-1/n)))
            if Bin_IP_est < 0:
                Bin_IP_est = IP
        else:
            Bin_IP_est = IP
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
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        
        denom = np.sqrt(spar_est_1) * np.sqrt(spar_est_2)
        if denom == 0:
            return 0.0
            
        Cosine_est = est_ip / denom
        return float(Cosine_est)
    