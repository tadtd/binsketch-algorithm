from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

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
        
        # Cache projection matrix P if not exists or dimensions changed
        if not hasattr(self, 'P') or self.P is None or self.P.shape != (n_features, k):
            rng = np.random.RandomState(seed=self.seed)
            buckets = rng.randint(0, k, size=n_features)
            row_indices = np.arange(n_features)
            col_indices = buckets
            data = np.ones(n_features, dtype=int)
            self.P = coo_matrix((data, (row_indices, col_indices)), shape=(n_features, k))

        X_sketch = X.dot(self.P)
        
        # Convert to dense and modulo 2
        X_sketch_dense = X_sketch.toarray().astype(int)
        X_sketch_binary = X_sketch_dense % 2
        
        return X_sketch_binary

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
        
        # For binary vectors, L2 norm = sqrt(count_nonzero) => but BCS doesn't preserve count exactly?
        # Actually BCS estimates IP(x, y). 
        # So IP(x, x) = ||x||^2.
        # Thus ||x|| = sqrt(IP(x, x))
        
        est_sq_norm1 = self.estimate_inner_product(sketch1, sketch1)
        est_sq_norm2 = self.estimate_inner_product(sketch2, sketch2)
        
        denom = np.sqrt(est_sq_norm1) * np.sqrt(est_sq_norm2)
        
        if denom == 0:
            return 0.0
            
        return est_ip / denom