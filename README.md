# BinSketch Algorithm Package

High-performance Python implementation of **BinSketch** and other binary sketching algorithms for efficient similarity estimation on high-dimensional binary data.

## Features

*   **Algorithms**: BinSketch, Binary Compressed Sensing (BCS), MinHash, SimHash.
*   **Performance**:
    *   **Dense Array Architecture**: Optimized for `numpy` vectorized operations.
    *   **GPU Acceleration**: Optional CUDA support via CuPy for massive speedups on large datasets.
    *   **Projection Caching**: Random projection matrices are generated once and cached, significantly speeding up repeated queries.
*   **Versatility**: Supports multiple similarity metrics:
    *   **Inner Product** (BinSketch, BCS)
    *   **Cosine Similarity** (BinSketch, SimHash, BCS)
    *   **Jaccard Similarity** (MinHash, BinSketch)
    *   **Hamming Distance** (SimHash, BinSketch)

## Installation

1.  **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2.  **Install dependencies**:
```bash
pip install -e .
```

3.  **Optional: Install GPU support** (requires NVIDIA GPU with CUDA):
```bash
# For CUDA 12.x
pip install -e ".[gpu]"

# For CUDA 11.x
pip install cupy-cuda11x

# Check installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

## Usage

### Basic Run (CPU)
```bash
python main.py --data_path ./data/nytimes_binary.npy --algo BinSketch --threshold 0.9 --similarity_score jaccard_similarity
```

### GPU-Accelerated Run
Enable GPU acceleration with the `--use_gpu` flag:
```bash
python main.py --data_path ./data/nytimes_binary.npy --algo BinSketch --threshold 0.9 --similarity_score jaccard_similarity --use_gpu
```

**Performance Gains**: GPU acceleration provides **5-20x speedup** on large datasets, especially beneficial for:
- Large document collections (>10,000 documents)
- High-dimensional data (>10,000 features)
- Multiple compression lengths and thresholds
- Matrix operations and similarity computations

### Advanced Examples

**1. GPU-accelerated experiment on NYTimes:**
```bash
python main.py --seed 42 --data_path ./data/nytimes_binary.npy --algo BinSketch BCS MinHash --threshold 0.1 0.2 0.3 0.4 0.5 0.7 0.8 0.9 --similarity_score jaccard_similarity --eval_metric minus_log_mse --use_gpu
```

**2. CPU vs GPU comparison:**
```bash
# CPU baseline
time python main.py --data_path ./data/nytimes_binary.npy --algo BinSketch --threshold 0.9 --similarity_score cosine_similarity

# GPU accelerated
time python main.py --data_path ./data/nytimes_binary.npy --algo BinSketch --threshold 0.9 --similarity_score cosine_similarity --use_gpu
```

## GPU Requirements

- **NVIDIA GPU** with CUDA Compute Capability 6.0 or higher
- **CUDA Toolkit** 11.x or 12.x installed
- **CuPy** compatible with your CUDA version

### Troubleshooting GPU

If GPU acceleration fails, the code will automatically fall back to CPU. Common issues:

1. **CuPy not installed**: Install with `pip install -e ".[gpu]"`
2. **CUDA version mismatch**: Ensure CuPy matches your CUDA version
3. **Out of memory**: Reduce batch size or use smaller compression lengths
4. **No CUDA device**: Verify GPU with `nvidia-smi`

## Project Structure

```
binsketch-algorithm/
├── src/
│   ├── models/
│   │   ├── base.py          # Abstract base class
│   │   ├── binsketch.py     # BinSketch (GPU-accelerated)
│   │   ├── bcs.py           # BCS (GPU-accelerated)
│   │   ├── minhash.py       # MinHash (GPU-accelerated)
│   │   └── simhash.py       # SimHash (GPU-accelerated)
│   ├── metric.py            # Evaluation metrics (GPU-enabled)
│   ├── similarity_scores.py # Similarity functions (GPU-enabled)
│   └── gpu_utils.py         # GPU/CPU compatibility layer
├── main.py                  # Main CLI experiment runner
└── README.md                # This file
```

## Algorithm Details

| Algorithm | Best For | Implementation Notes |
| :--- | :--- | :--- |
| **BinSketch** | Inner Product / Cosine | Uses bias correction for sparsity. GPU-accelerated. |
| **BCS** | Inner Product | Simple random projection. GPU-accelerated. |
| **SimHash** | Cosine / Hamming | Random hyperplane projection. GPU-accelerated. |
| **MinHash** | Jaccard | Permutation-based sketching. GPU-accelerated. |