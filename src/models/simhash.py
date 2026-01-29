from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix
from ..gpu_utils import (
    get_array_module, to_gpu, to_cpu,
    create_random_state, GPUConfig
)
from tqdm import tqdm

class SimHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.R = None

    def mapping(self, X: csr_matrix, k: int, return_gpu: bool = False) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional SimHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
            return_gpu: If True and GPU is enabled, return GPU array without transferring to CPU.
        """
        _, n = X.shape
        use_gpu = GPUConfig.is_enabled()
        
        # Transfer to GPU if enabled
        if use_gpu:
            X = to_gpu(X)
        
        if self.R is None or self.R.shape != (n, k):
            rng = create_random_state(self.seed, use_gpu)
            xp = get_array_module()
            if use_gpu:
                self.R = rng.randn(n, k).astype(xp.float32)
            else:
                self.R = rng.randn(n, k).astype(np.float64)
            
        predictions = X.dot(self.R)
        xp = get_array_module(predictions if hasattr(predictions, 'shape') else predictions.toarray())
        
        if hasattr(predictions, 'toarray'):
            predictions = predictions.toarray()
        
        if use_gpu:
            sketch = (predictions >= 0).astype(xp.float32)
        else:
            sketch = (predictions >= 0).astype(np.int8)
        
        # Return GPU array if requested, otherwise transfer to CPU
        if return_gpu and use_gpu:
            return sketch
        
        result = to_cpu(sketch)
        if use_gpu:
            result = result.astype(np.int8)
        return result
    
    def estimate_hamming_distance(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Hamming distance.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Hamming distance estimation.")
        
        xp = get_array_module(sketch1)
        
        # Efficient hamming using XOR
        diff = xp.bitwise_xor(sketch1, sketch2)
        result = xp.count_nonzero(diff)
        
        return float(to_cpu(result))

    def estimate_cosine_similarity(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimates Cosine similarity.
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Cosine similarity estimation.")
        
        xp = get_array_module(sketch1)
        
        n = sketch1.shape[-1]
        Bin_Ham_est = self.estimate_hamming_distance(sketch1, sketch2)
        ratio = Bin_Ham_est / n
        Bin_Cosine_est = xp.cos(xp.pi * ratio) if hasattr(xp, 'pi') else np.cos(np.pi * ratio)
        
        return float(to_cpu(Bin_Cosine_est))
    
    def estimate_hamming_distance_batch(self, sketches: np.ndarray, pairs: list) -> np.ndarray:
        """
        Estimates Hamming distances for multiple pairs efficiently using GPU vectorization.
        
        Args:
            sketches: Array of sketches (n_samples, sketch_dim) - can be on GPU or CPU
            pairs: List of (i, j) tuples indicating which pairs to estimate
            
        Returns:
            Array of estimated Hamming distances for each pair
        """
        xp = get_array_module(sketches)
        
        # Extract indices
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Batch XOR and count
        diff = xp.bitwise_xor(sketches_i.astype(xp.int32), sketches_j.astype(xp.int32))
        result = xp.count_nonzero(diff, axis=1)
        
        return to_cpu(result).astype(np.float32)
    
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
        n = sketches.shape[-1]
        
        # Extract indices
        indices_i = xp.array([i for i, j in pairs], dtype=xp.int32)
        indices_j = xp.array([j for i, j in pairs], dtype=xp.int32)
        
        # Batch gather sketches
        sketches_i = sketches[indices_i]
        sketches_j = sketches[indices_j]
        
        # Batch Hamming distances
        diff = xp.bitwise_xor(sketches_i.astype(xp.int32), sketches_j.astype(xp.int32))
        hamming_dists = xp.count_nonzero(diff, axis=1)
        
        # Vectorized cosine estimation
        ratio = hamming_dists / n
        cosine_est = xp.cos(xp.pi * ratio) if hasattr(xp, 'pi') else np.cos(np.pi * ratio)
        
        return to_cpu(cosine_est).astype(np.float32)