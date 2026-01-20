from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix

class SimHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.R = None

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional SimHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n = X.shape
        
        # Cache random projection matrix R
        if self.R is None or self.R.shape != (n, k):
            self.R = np.random.RandomState(seed=self.seed).randn(n, k)
            
        predictions = X.dot(self.R)
        sketch = (predictions >= 0).astype(np.int8)
        return sketch
    
    def estimate_hamming_distance(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Hamming distance.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Hamming distance estimation.")
        
        # Efficient hamming using XOR
        diff = np.bitwise_xor(sketch1, sketch2)
        return float(np.count_nonzero(diff))

    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Cosine similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        n = sketch1.shape[-1]
        Bin_Ham_est = self.estimate_hamming_distance(sketch1, sketch2)
        ratio = Bin_Ham_est / n
        Bin_Cosine_est = np.cos(np.pi * ratio)
        return float(Bin_Cosine_est)