from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix
from ..gpu_utils import (
    get_array_module, to_cpu, to_gpu, create_random_state, GPUConfig, zeros
)
from tqdm import tqdm

class MinHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.BIG_PRIME = 2147483647 # Mersenne Prime 2^31 - 1, safe for int32
        self.coeffs = None
        self.k = None

    def mapping(self, X: csr_matrix, k: int, return_gpu: bool = False) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional MinHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
            return_gpu: If True and GPU is enabled, return GPU array without transferring to CPU.
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
        
        # Return GPU array if requested, otherwise transfer to CPU
        if return_gpu and use_gpu:
            return result
        
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
        
        xp = get_array_module(sketch1)
        
        sig1 = sketch1.ravel()
        sig2 = sketch2.ravel()
        
        matches = xp.sum(sig1 == sig2)
        jaccard_sim = matches / len(sig1)
        
        # Convert to Python float (handles both CPU and GPU)
        return float(to_cpu(jaccard_sim))
    
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
        
        # Batch comparison
        matches = xp.sum(sketches_i == sketches_j, axis=1)
        k = sketches.shape[1]
        jaccard_sims = matches / k
        
        return to_cpu(jaccard_sims).astype(np.float32)
    
    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates cosine similarity between two MinHash sketches.
        
        Args:
            sketch1: First MinHash sketch.
            sketch2: Second MinHash sketch.
        
        Returns:
            Estimated cosine similarity (between -1 and 1).
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for cosine similarity estimation.")
        
        xp = get_array_module(sketch1)
        
        sig1 = sketch1.ravel().astype(xp.float64)
        sig2 = sketch2.ravel().astype(xp.float64)
        
        # Compute dot product
        dot_product = xp.sum(sig1 * sig2)
        
        # Compute norms (using float64 to avoid overflow)
        norm1_sq = xp.sum(sig1 ** 2)
        norm2_sq = xp.sum(sig2 ** 2)
        
        # Check for invalid values before sqrt
        if not xp.isfinite(norm1_sq) or not xp.isfinite(norm2_sq) or norm1_sq <= 0 or norm2_sq <= 0:
            return 0.0
        
        norm1 = xp.sqrt(norm1_sq)
        norm2 = xp.sqrt(norm2_sq)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Convert to Python float (handles both CPU and GPU)
        return float(to_cpu(cosine_sim))
    
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
        
        # Compute dot products
        dot_products = xp.sum(sketches_i * sketches_j, axis=1)
        
        # Compute norms
        norms_i = xp.sqrt(xp.sum(sketches_i ** 2, axis=1))
        norms_j = xp.sqrt(xp.sum(sketches_j ** 2, axis=1))
        
        # Avoid division by zero
        denominators = norms_i * norms_j
        denominators = xp.maximum(denominators, 1e-10)
        
        cosine_sims = dot_products / denominators
        
        return to_cpu(cosine_sims).astype(np.float32)