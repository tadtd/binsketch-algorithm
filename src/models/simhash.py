from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix

class SimHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def mapping(self, X: csr_matrix, k: int) -> csr_matrix:
        """
        Projects high-dimensional binary data into a lower-dimensional SimHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n = X.shape
        R = np.random.RandomState(seed=self.seed).randn(n, k)
        predictions = X@R
        sketch = (predictions >= 0).astype(X.dtype)
        return csr_matrix(sketch)
    
    def estimate_hamming_distance(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Hamming distance.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Hamming distance estimation.")
        diff = sketch1 - sketch2
        return float(np.abs(diff.data).sum())

    def estimate_cosine_similarity(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Cosine similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        _, n = sketch1.shape
        Bin_Ham_est = self.estimate_hamming_distance(sketch1, sketch2)
        ratio = Bin_Ham_est / n
        Bin_Cosine_est = np.cos(np.pi * ratio)
        return float(Bin_Cosine_est)