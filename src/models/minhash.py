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
        
        For binary vectors, we use the relationship between Jaccard and cosine similarity.
        MinHash estimates Jaccard similarity J = |A ∩ B| / |A ∪ B|.
        
        For binary vectors: cos(a,b) = |A ∩ B| / sqrt(|A| * |B|)
        
        We can derive: if J = intersection / union, then:
        intersection = J * union = J * (|A| + |B| - intersection)
        So: intersection = J * (|A| + |B|) / (1 + J)
        
        However, since MinHash doesn't preserve cardinality information,
        we use an approximation: cos ≈ 2*J / (1 + J) for roughly equal-sized sets.
        
        Args:
            sketch1: First MinHash sketch.
            sketch2: Second MinHash sketch.
        
        Returns:
            Estimated cosine similarity (between 0 and 1).
        """
        # First estimate Jaccard similarity
        jaccard_sim = self.estimate_jaccard_similarity(sketch1, sketch2)
        
        # Convert Jaccard to cosine approximation
        # For binary vectors with similar sparsity: cos ≈ 2*J / (1 + J)
        # This is derived from the relationship between intersection and union
        if jaccard_sim <= 0:
            return 0.0
        
        # More accurate approximation: cos ≈ sqrt(J) for many practical cases
        # Or use: cos ≈ 2*J / (1+J) which gives a reasonable mapping
        cosine_sim = 2 * jaccard_sim / (1 + jaccard_sim)
        
        return float(cosine_sim)
    
    def estimate_cosine_similarity_batch(self, sketches: np.ndarray, pairs: list) -> np.ndarray:
        """
        Estimates cosine similarities for multiple pairs efficiently using GPU vectorization.
        
        Uses the relationship between Jaccard and cosine similarity.
        
        Args:
            sketches: Array of sketches (n_samples, sketch_dim) - can be on GPU or CPU
            pairs: List of (i, j) tuples indicating which pairs to estimate
            
        Returns:
            Array of estimated cosine similarities for each pair
        """
        xp = get_array_module(sketches)
        
        # First compute Jaccard similarities using existing batch method
        jaccard_sims = self.estimate_jaccard_similarity_batch(sketches, pairs)
        
        # Convert to GPU array if needed for vectorized operations
        use_gpu = GPUConfig.is_enabled()
        if use_gpu:
            jaccard_sims_xp = xp.asarray(jaccard_sims)
        else:
            jaccard_sims_xp = jaccard_sims
        
        # Convert Jaccard to cosine: cos ≈ 2*J / (1 + J)
        cosine_sims = 2 * jaccard_sims_xp / (1 + jaccard_sims_xp + 1e-10)
        
        # Handle edge case where Jaccard is 0
        cosine_sims = xp.where(jaccard_sims_xp <= 0, 0.0, cosine_sims)
        
        return to_cpu(cosine_sims).astype(np.float32)