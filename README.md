# BinSketch Algorithm Package

High-performance Python implementation of **BinSketch** and other binary sketching algorithms for efficient similarity estimation on high-dimensional binary data.

## 🚀 Features

*   **Algorithms**: BinSketch, Binary Compressed Sensing (BCS), MinHash, SimHash.
*   **Performance**:
    *   **Dense Array Architecture**: Optimized for `numpy` vectorized operations.
    *   **Projection Caching**: Random projection matrices are generated once and cached, significantly speeding up repeated queries.
*   **Versatility**: Supports multiple similarity metrics:
    *   **Inner Product** (BinSketch, BCS)
    *   **Cosine Similarity** (BinSketch, SimHash, BCS)
    *   **Jaccard Similarity** (MinHash, BinSketch)
    *   **Hamming Distance** (SimHash, BinSketch)

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tadtd/binsketch-algorithm
    cd binsketch-algorithm
    ```

2.  **Install dependencies**:
    This project relies on `numpy` and `scipy`.
    ```bash
    pip install numpy scipy
    ```
    *(Optional: dedicated environment recommended)*

## 🛠 Usage

The main entry point is `test_pipeline.py`. It provides a flexible CLI to test and compare different algorithms and metrics.

### Basic Run
Run the full test suite on all algorithms and metrics:
```bash
python test_pipeline.py
```

### Select Algorithms
Test specific algorithms using the `--algo` flag:
```bash
# Only BinSketch
python test_pipeline.py --algo BinSketch

# Compare BinSketch and BCS
python test_pipeline.py --algo BinSketch BCS
```
*Supported Algorithms*: `BinSketch`, `BCS`, `MinHash`, `SimHash`.

### Select Metrics
Test specific similarity measures using the `--metric` flag:
```bash
# Only Cosine Similarity
python test_pipeline.py --metric cosine

# Compare Jaccard and Hamming
python test_pipeline.py --metric jaccard hamming
```
*Supported Metrics*: `inner_product`, `cosine`, `jaccard`, `hamming`.

### Advanced Examples

**1. Compare BinSketch vs SimHash on Cosine Similarity (their "Home Ground"):**
```bash
python test_pipeline.py --algo BinSketch SimHash --metric cosine
```

**2. Test BinSketch's versatility across all metrics:**
```bash
python test_pipeline.py --algo BinSketch --metric inner_product cosine jaccard hamming
```

**3. Run on specific dataset sizes:**
By default, the pipeline runs a suite of tests ranging from Tiny (100x100) to Very Large (500x1,000,000). You can modify the `test_configs` in `test_pipeline.py` to customize this.

## 📂 Project Structure

```
binsketch-algorithm/
├── src/
│   ├── models/
│   │   ├── base.py          # Abstract base class
│   │   ├── binsketch.py     # BinSketch implementation
│   │   ├── bcs.py           # BCS implementation
│   │   ├── minhash.py       # MinHash implementation
│   │   └── simhash.py       # SimHash implementation
│   ├── metric.py            # MSE, MAE, F1 functions
│   └── similarity_scores.py # Ground truth calculation utils
├── test_pipeline.py         # Main CLI testing tool
└── README.md                # This file
```

## 📝 Algorithm Details

| Algorithm | Best For | Implementation Notes |
| :--- | :--- | :--- |
| **BinSketch** | Inner Product / Cosine | Uses bias correction for sparsity. Supports all metrics. |
| **BCS** | Inner Product | Simple random projection. Fast but less robust on skewed data. |
| **SimHash** | Cosine / Hamming | Random hyperplane projection. Standard LSH for Cosine. |
| **MinHash** | Jaccard | Permutation-based sketching. Standard for Set Similarity. |