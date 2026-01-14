from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class BinarySchemaCompression(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def mapping(self, X: csr_matrix, k: int) -> csr_matrix:
        """
        Projects high-dimensional binary data into a lower-dimensional binary sketch.
        
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        """
        _, n_features = X.shape
        rng = np.random.RandomState(seed=self.seed)
        buckets = rng.randint(0, k, size=n_features)
        row_indices = np.arange(n_features)
        col_indices = buckets
        data = np.ones(n_features, dtype=int)

        P = coo_matrix((data, (row_indices, col_indices)), shape=(n_features, k)) # Projection matrix
        X_sketch = X.dot(P)
        X_sketch.data = X_sketch.data.astype(int) % 2
        X_sketch.eliminate_zeros()
        
        return csr_matrix(X_sketch)

    def estimate_inner_product(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates inner product.
        """
        return float(sketch1.multiply(sketch2).sum())

    def estimate_hamming_distance(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Hamming distance.
        """
        diff = sketch1 - sketch2
        return float(np.abs(diff.data).sum())

    def estimate_jaccard_similarity(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Jaccard similarity: Intersection / Union
        Union = Hamming + Intersection
        """
        est_ip = self.estimate_inner_product(sketch1, sketch2)
        est_hamming = self.estimate_hamming_distance(sketch1, sketch2)
        
        union = est_hamming + est_ip
        
        if union == 0:
            return 0.0
            
        return est_ip / union