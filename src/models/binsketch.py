from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class BinSketch(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def _estimate_sparsity(self, a: csr_matrix) -> float:
        """Estimates the original vector's sparsity from its sketch."""
        _, n = a.shape
        spar_a = a.nnz 
        if spar_a / n < 1:
            spar_est_a = np.round(np.log(1-spar_a/n) / np.log(1-1/n))
        else:
            spar_est_a = spar_a
        return spar_est_a

    def mapping(self, X: csr_matrix, k: int) -> csr_matrix:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n = X.shape
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
        
        P = csc_matrix((data, (row_indices, col_indices)), shape=(n, k))
        X_sketch = X.dot(P)
        X_sketch.data = np.where(X_sketch.data > 0, 1, 0).astype(X.dtype)
        return csr_matrix(X_sketch)
    
    def estimate_inner_product(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates inner product.
        """
        _, n = sketch1.shape
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        IP = int(sketch1@(sketch2.T).toarray()[0,0])
        val = (1-1/n)**spar_est_1 + (1-1/n)**spar_est_2 - 1 + IP/n
        if val > 0:
            Bin_IP_est = np.round(spar_est_1 + spar_est_2 - (np.log(val)/np.log(1-1/n)))
            if Bin_IP_est < 0:
                Bin_IP_est = IP
        else:
            Bin_IP_est = IP
        return Bin_IP_est
    
    def estimate_hamming_distance(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Hamming distance.
        """
        spar_est_1 = self._estimate_sparsity(sketch1)
        spar_est_2 = self._estimate_sparsity(sketch2)
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        Bin_Ham_est = spar_est_1 + spar_est_2 - 2*est_ip
        return Bin_Ham_est
    
    def estimate_jaccard_similarity(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Jaccard similarity.
        """
        raise NotImplementedError("Jaccard similarity estimation not implemented yet.")
    
    def estimate_cosine_similarity(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Cosine similarity.
        """
        raise NotImplementedError("Cosine similarity estimation not implemented yet.")
    