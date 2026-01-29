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

    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        """
        Projects high-dimensional binary data into a lower-dimensional SimHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
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