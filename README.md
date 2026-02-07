# BinSketch Algorithm

Implementation and evaluation of binary sketching algorithms for efficient similarity estimation on high-dimensional sparse binary data.

Based on the paper: [*Efficient Sketching Algorithm for Sparse Binary Data*](https://arxiv.org/pdf/1910.04658) (Pratap et al., 2019).

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management.

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup the project

```bash
git clone https://github.com/tadtd/binsketch-algorithm.git
cd binsketch-algorithm

# Create virtual environment and install dependencies
uv sync
```
---

## How to Run

### 1. Download and process data

Download raw datasets from Google Drive:

```bash
uv run python -c "
import gdown
gdown.download_folder(
    'https://drive.google.com/drive/folders/1ARBY9cIGj_jigi5Y88CtUy-GMj2clrXj',
    output='./raw',
    quiet=False,
    use_cookies=False
)
"
```

Convert raw data to `.npy` format:

```bash
uv run python scripts/convert.py
```

### 2. Run experiments

**Experiment 1 – Accuracy of similarity estimation (MSE)**

```bash
uv run python main.py --experiment 1 \
    --algo BinSketch BCS \
    --data_path ./data/nytimes_binary.npy \
    --seed 42 \
    --threshold 120 150 180 200 220 250 270 300 \
    --similarity_score inner_product \
    --eval_metric mse \
```

**Experiment 2 – Retrieval performance (Precision/Recall/F1)**

```bash
uv run python main.py --experiment 2 \
    --algo BinSketch BCS MinHash \
    --data_path ./data/nytimes_binary.npy \
    --train_ratio .9 \
    --seed 42 \
    --threshold .1 .2 .4 .5 .6 .7 .85 .95 \
    --similarity_score jaccard_similarity \
    --retrieval_metric f1 \
```

Other datasets: `bbc_binary.npy`, `enron_binary.npy`, `kos_binary.npy`

**NOTE:** To leverage the power of GPU (e.g. Kaggle), you can view the notebook for interactive runs.
Results and plots are saved to `results/experiment1/` and `results/experiment2/`.

---

## Folder structure

```
binsketch-algorithm/
├── bin_sketch.ipynb       # Jupyter notebook for interactive runs (e.g. Kaggle)
├── main.py                # CLI experiment runner
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions
├── README.md
│
├── data/                  # Processed datasets (.npy)
│   ├── bbc_binary.npy
│   ├── bbc_vocab.npy
│   ├── enron_binary.npy
│   ├── enron_vocab.npy
│   ├── kos_binary.npy
│   ├── kos_vocab.npy
│   ├── nytimes_binary.npy
│   └── nytimes_vocab.npy
│
├── raw/                   # Raw datasets (UCI bag-of-words format)
│   ├── bbc/
│   │   ├── docword.bbc.txt.gz
│   │   └── vocab.bbc.txt
│   ├── enron/
│   │   ├── docword.enron.txt.gz
│   │   └── vocab.enron.txt
│   ├── kos/
│   │   ├── docword.kos.txt.gz
│   │   └── vocab.kos.txt
│   └── nytimes/
│       ├── docword.nytimes.txt.gz
│       └── vocab.nytimes.txt
│
├── src/                   # Core implementation
│   ├── __init__.py
│   ├── gpu_utils.py
│   ├── metric.py
│   ├── similarity_scores.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── bcs.py
│       ├── binsketch.py
│       ├── minhash.py
│       └── simhash.py
│
├── experiment/            # Experiment runners
│   ├── __init__.py
│   ├── experiment1.py     # MSE-based accuracy evaluation
│   ├── experiment2.py    # Retrieval (Precision/Recall/F1)
│   └── utils.py
│
├── scripts/               # Data processing
│   ├── convert.py        # raw → .npy conversion
│   ├── filter_json.py
│   └── save_ground_truth.py
│
├── results/               # Output plots and results
│   ├── experiment1/
│   └── experiment2/
│
└── tests/
    └── test_pipeline.py
```

---

## Reference

Pratap, R., Bera, D., & Revanuru, K. (2019). *Efficient Sketching Algorithm for Sparse Binary Data.* arXiv:1910.04658. [https://arxiv.org/pdf/1910.04658](https://arxiv.org/pdf/1910.04658)
