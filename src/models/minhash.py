from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix
from ..gpu_utils import (
    get_array_module, to_cpu, create_random_state, GPUConfig, zeros
)

class MinHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.BIG_PRIME = 2147483647 # Mersenne Prime 2^31 - 1, safe for int32
        self.coeffs = None
        self.k = None

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional MinHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        Returns:
            Sketch matrix where each row contains k MinHash signatures.
        """
        n_samples, n_features = X.shape
        
        use_gpu = GPUConfig.is_enabled()
        xp = get_array_module()
        
        # Initialize hashing coefficients if new k or first run
        if self.coeffs is None or self.k != k:
            self.k = k
            rng = create_random_state(self.seed, use_gpu)
            # Generate k pairs of coefficients (a, b)
            # Use float64 for GPU compatibility in hash computations
            if use_gpu:
                self.coeffs = rng.randint(1, self.BIG_PRIME, size=(k, 2)).astype(xp.float64)
            else:
                self.coeffs = rng.randint(1, self.BIG_PRIME, size=(k, 2)).astype(xp.int64)

        sketches = []
        
        # Vectorized hashing for MinHash
        # We need to compute h(x) = (a*x + b) % PRIME for all non-zero x
        # And find the min value for each of the k hash functions.
        
        for i in range(n_samples):
            # Get indices of non-zero elements (the "Set")
            row = X.getrow(i)
            indices = row.indices # raw indices array
            
            if indices.size == 0:
                sketches.append(zeros(k, dtype=xp.float32 if use_gpu else int, use_gpu=use_gpu))
                continue
            
            if use_gpu:
                indices = xp.asarray(indices, dtype=xp.float64)
            
            a = self.coeffs[:, 0].reshape(-1, 1)
            b = self.coeffs[:, 1].reshape(-1, 1)
            
            distinct_hashes = (a * indices + b) % self.BIG_PRIME
            min_hashes = xp.min(distinct_hashes, axis=1)
            sketches.append(min_hashes)
        
        result = xp.array(sketches)
        result = to_cpu(result)
        if use_gpu:
            result = result.astype(np.int32)
        return result
    
    def estimate_jaccard_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Jaccard similarity between two MinHash sketches.
        Args:
            sketch1: First MinHash sketch.
            sketch2: Second MinHash sketch.
        Returns:
            Estimated Jaccard similarity (between 0 and 1).
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Jaccard similarity estimation.")
        
        sig1 = sketch1.ravel()
        sig2 = sketch2.ravel()
        
        matches = np.sum(sig1 == sig2)
        jaccard_sim = matches / len(sig1)
        
        return float(jaccard_sim)