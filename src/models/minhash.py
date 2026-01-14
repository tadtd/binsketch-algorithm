from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix

class MinHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional MinHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        raise NotImplementedError("Mapping method not implemented yet.")
    
    def estimate_jaccard_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Jaccard similarity.
        """
        raise NotImplementedError("Jaccard similarity estimation not implemented yet.")