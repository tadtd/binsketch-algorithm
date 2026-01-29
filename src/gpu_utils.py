"""GPU utilities with automatic fallback to CPU."""
import os
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    from cupyx.scipy.sparse import csc_matrix as gpu_csc_matrix
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    gpu_csr_matrix = None
    gpu_csc_matrix = None
    GPU_AVAILABLE = False

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


class GPUConfig:
    """Global GPU configuration."""
    _use_gpu = False
    _gpu_initialized = False
    
    @classmethod
    def enable_gpu(cls):
        """Enable GPU acceleration if available."""
        if not GPU_AVAILABLE:
            warnings.warn(
                "GPU acceleration requested but CuPy is not installed. "
                "Install with: pip install cupy-cuda12x (replace 12x with your CUDA version). "
                "Falling back to CPU."
            )
            cls._use_gpu = False
            return False
        
        try:
            # Test GPU availability
            cp.cuda.Device(0).compute_capability
            cls._use_gpu = True
            cls._gpu_initialized = True
            print("✓ GPU acceleration enabled")
            return True
        except Exception as e:
            warnings.warn(f"GPU initialization failed: {e}. Falling back to CPU.")
            cls._use_gpu = False
            return False
    
    @classmethod
    def disable_gpu(cls):
        """Disable GPU acceleration."""
        cls._use_gpu = False
        if cls._gpu_initialized:
            print("✓ GPU acceleration disabled")
    
    @classmethod
    def is_enabled(cls):
        """Check if GPU acceleration is enabled."""
        return cls._use_gpu and GPU_AVAILABLE


def get_array_module(arr=None):
    """
    Get the appropriate array module (numpy or cupy) for the given array.
    If arr is None, returns the module based on GPU config.
    """
    if arr is None:
        return cp if GPUConfig.is_enabled() else np
    
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def get_sparse_module(arr=None):
    """Get the appropriate sparse matrix module."""
    if arr is None:
        if GPUConfig.is_enabled():
            from cupyx.scipy import sparse as cusp
            return cusp
        else:
            from scipy import sparse
            return sparse
    
    if GPU_AVAILABLE:
        from cupyx.scipy import sparse as cusp
        if isinstance(arr, (cusp.csr_matrix, cusp.csc_matrix)):
            return cusp
    
    from scipy import sparse
    return sparse


def to_gpu(arr):
    """Transfer array/matrix to GPU."""
    if not GPUConfig.is_enabled():
        return arr
    
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    elif isinstance(arr, (csr_matrix, csc_matrix)):
        # Check if data needs conversion BEFORE creating GPU sparse matrix
        needs_conversion = arr.data.dtype not in [np.bool_, np.float32, np.float64, np.complex64, np.complex128]
        
        if needs_conversion:
            print(f"  [GPU] Converting sparse matrix data from {arr.data.dtype} to float32")
            # Create a CPU sparse matrix with converted dtypes first
            if isinstance(arr, csr_matrix):
                arr_converted = csr_matrix(
                    (arr.data.astype(np.float32), arr.indices.astype(np.int32), arr.indptr.astype(np.int32)),
                    shape=arr.shape
                )
                gpu_sparse = gpu_csr_matrix(arr_converted)
            else:
                arr_converted = csc_matrix(
                    (arr.data.astype(np.float32), arr.indices.astype(np.int32), arr.indptr.astype(np.int32)),
                    shape=arr.shape
                )
                gpu_sparse = gpu_csc_matrix(arr_converted)
        else:
            # Data is already compatible, transfer directly
            gpu_sparse = gpu_csr_matrix(arr) if isinstance(arr, csr_matrix) else gpu_csc_matrix(arr)
        
        return gpu_sparse
    elif GPU_AVAILABLE and isinstance(arr, (cp.ndarray, gpu_csr_matrix, gpu_csc_matrix)):
        return arr  # Already on GPU
    else:
        raise TypeError(f"Unsupported type for GPU transfer: {type(arr)}")


def to_cpu(arr):
    """Transfer array/matrix to CPU."""
    if GPU_AVAILABLE:
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        elif isinstance(arr, (gpu_csr_matrix, gpu_csc_matrix)):
            return arr.get()
    
    return arr  # Already on CPU or not a GPU array


def ensure_same_device(arr1, arr2):
    """Ensure both arrays are on the same device (CPU or GPU)."""
    xp = get_array_module(arr1)
    
    # If arr1 is on GPU but arr2 is not, move arr2 to GPU
    if xp.__name__ == 'cupy' and not isinstance(arr2, cp.ndarray):
        return arr1, to_gpu(arr2)
    
    # If arr1 is on CPU but arr2 is on GPU, move arr2 to CPU
    if xp.__name__ == 'numpy' and GPU_AVAILABLE and isinstance(arr2, cp.ndarray):
        return arr1, to_cpu(arr2)
    
    return arr1, arr2


def create_random_state(seed, use_gpu=None):
    """Create random state for CPU or GPU."""
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()
    
    if use_gpu and GPU_AVAILABLE:
        return cp.random.RandomState(seed)
    else:
        return np.random.RandomState(seed)


def asnumpy(arr):
    """Convert array to numpy, regardless of source device."""
    return to_cpu(arr)


def zeros(shape, dtype=None, use_gpu=None):
    """Create zeros array on CPU or GPU."""
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()
    
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None, use_gpu=None):
    """Create ones array on CPU or GPU."""
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()
    
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    return xp.ones(shape, dtype=dtype)


def concatenate(arrays, axis=0, use_gpu=None):
    """Concatenate arrays on CPU or GPU."""
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()
    
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    return xp.concatenate(arrays, axis=axis)


def arange(*args, use_gpu=None, dtype=None):
    """Create arange array on CPU or GPU."""
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()
    
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    result = xp.arange(*args)
    
    # Apply dtype if specified
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


def batch_gather(arr, indices, use_gpu=None):
    """
    Efficiently gather rows from array based on indices.
    
    Args:
        arr: Array to gather from (n_samples, features)
        indices: Indices to gather (n_pairs,)
        use_gpu: Whether to use GPU (None = auto-detect)
    
    Returns:
        Gathered array (n_pairs, features)
    """
    xp = get_array_module(arr)
    return arr[indices]


def ensure_gpu(arr):
    """
    Ensure array is on GPU if GPU is enabled.
    
    Args:
        arr: Input array (can be CPU or GPU)
    
    Returns:
        GPU array if GPU is enabled, otherwise unchanged
    """
    if GPUConfig.is_enabled() and GPU_AVAILABLE:
        xp = get_array_module(arr)
        if xp.__name__ == 'numpy':
            return to_gpu(arr)
    return arr


def compute_pairwise_distances(sketches, metric='euclidean', chunk_size=1000):
    """
    Compute pairwise distances efficiently using GPU.
    
    Args:
        sketches: Array of sketches (n_samples, sketch_dim)
        metric: Distance metric ('euclidean', 'hamming', etc.)
        chunk_size: Process in chunks to manage memory
    
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    xp = get_array_module(sketches)
    n_samples = sketches.shape[0]
    
    if metric == 'hamming':
        # For binary sketches, hamming = XOR then count
        # Process in chunks to avoid memory issues
        distances = xp.zeros((n_samples, n_samples), dtype=xp.float32)
        
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk_i = sketches[i:end_i]
            
            for j in range(0, n_samples, chunk_size):
                end_j = min(j + chunk_size, n_samples)
                chunk_j = sketches[j:end_j]
                
                # Broadcasting: (chunk_i_size, 1, dim) vs (1, chunk_j_size, dim)
                diff = chunk_i[:, xp.newaxis, :] != chunk_j[xp.newaxis, :, :]
                distances[i:end_i, j:end_j] = xp.sum(diff, axis=2)
        
        return distances
    else:
        raise NotImplementedError(f"Metric {metric} not implemented yet")

