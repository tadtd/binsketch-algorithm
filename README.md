# BinSketch Algorithm Experiments

This repository provides a comprehensive implementation and evaluation framework for binary sketching algorithms designed for efficient similarity estimation on high-dimensional binary data.

## About This Repository

BinSketch algorithms are essential for handling large-scale binary datasets where computing exact similarities is computationally expensive. This repository implements several key algorithms and provides:

- **Reproducible experiments** comparing algorithm performance
- **Multiple similarity metrics** (inner product, cosine similarity, Jaccard similarity)
- **Real-world datasets** (BBC, Enron, KOS, NYTimes)
- **Comprehensive evaluation** covering both accuracy and retrieval tasks
- **Easy-to-use interface** for running experiments locally or on Kaggle

The implementation focuses on efficiency and accuracy, providing researchers and practitioners with tools to evaluate and compare binary sketching methods on their own datasets.

## Experiments
- **Experiment 1**: Accuracy of similarity estimation (MSE-based evaluation)
- **Experiment 2**: Retrieval performance (Precision/Recall/F1/Accuracy)

## How to Run (Kaggle)

1. Upload the provided notebook (e.g. `bin_sketch.ipynb`) to [Kaggle Notebooks](https://www.kaggle.com/code).
2. (If needed) Add your GitHub token to Kaggle Secrets for private repo access.
3. Run all cells. The notebook will:
   - Clone the code
   - Download and process datasets
   - Run Experiment 1 (accuracy) and Experiment 2 (ranking)
   - Save results/plots to the output directory
4. Download results from the sidebar if needed.

## Customizing
- Edit variables in the notebook cells to change dataset, metric, algorithms, or experiment type.
- All commands and workflow are already scripted in the notebook.

## Example: Run Experiment 1 (in notebook)
```python
!python main.py --experiment 1 --algo BinSketch BCS --data_path ./data/nytimes_binary.npy --ground_truth_path ground_truth_exp1_nytimes_inner_product.json --seed 42 --threshold 120 150 180 200 220 250 270 300 --similarity_score inner_product --eval_metric mse --use_gpu
```

## Example: Run Experiment 2 (in notebook)
```python
!python main.py --experiment 2 --algo BinSketch BCS MinHash --data_path ./data/nytimes_binary.npy --ground_truth_path ground_truth_exp2_nytimes_jaccard_similarity.json --train_ratio .9 --seed 42 --threshold .1 .2 .4 .5 .6 .7 .85 .95 --similarity_score jaccard_similarity --retrieval_metric f1 --use_gpu
```

**Just upload and run the notebook—no manual setup needed!**

## How to Run Experiments on Your Own Computer

1. **Clone the repository**
   ```bash
   git clone https://github.com/tadtd/binsketch-algorithm.git
   cd binsketch-algorithm
   ```

2. **Create and activate a virtual environment (optional)**
   
   **Option A: Using uv (Recommended - faster)**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Create and activate virtual environment
   uv venv venv
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
   
   **Option B: Using standard venv**
   ```bash
   python -m venv venv
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **If using uv:**
   ```bash
   uv sync
   ```
   
   **If using standard pip:**
   ```bash
   pip install -e .
   pip install gdown
   ```

4. **Download and preprocess the datasets**
   - Download the raw data from Google Drive:
     ```python
     import gdown
     gdown.download_folder('https://drive.google.com/drive/folders/1ARBY9cIGj_jigi5Y88CtUy-GMj2clrXj', output='./raw', quiet=False, use_cookies=False)
     ```
   - Convert the raw data to .npy format:
     ```bash
     python scripts/convert.py
     ```

5. **Run Experiment 1 (Accuracy Estimation)**
   ```bash
   # Generate ground truth
   python scripts/save_ground_truth.py --experiment 1 --data_path ./data/nytimes_binary.npy --similarity_score inner_product --seed 42
   
   # Run experiment
   python main.py --experiment 1 --algo BinSketch BCS --data_path ./data/nytimes_binary.npy --ground_truth_path ground_truth_exp1_nytimes_inner_product.json --seed 42 --threshold 120 150 180 200 220 250 270 300 --similarity_score inner_product --eval_metric mse
   ```

6. **Run Experiment 2 (Retrieval/Ranking)**
   ```bash
   # Generate ground truth
   python scripts/save_ground_truth.py --experiment 2 --data_path ./data/nytimes_binary.npy --similarity_score jaccard_similarity --train_ratio .9 --seed 42
   
   # Run experiment
   python main.py --experiment 2 --algo BinSketch BCS MinHash --data_path ./data/nytimes_binary.npy --ground_truth_path ground_truth_exp2_nytimes_jaccard_similarity.json --train_ratio .9 --seed 42 --threshold .1 .2 .4 .5 .6 .7 .85 .95 --similarity_score jaccard_similarity --retrieval_metric f1
   ```

7. **Check the output directory for results and plots.**

## Project Structure

```
binsketch-algorithm/
├── bin_sketch.ipynb         # Main Kaggle notebook
├── main.py                  # CLI experiment runner
├── pyproject.toml           # Project metadata
├── uv.lock                  # uv lock file
├── README.md                # This file
├── data/                    # Processed datasets (auto-downloaded)
│   ├── bbc_binary.npy
│   ├── bbc_vocab.npy
│   ├── enron_binary.npy
│   ├── enron_vocab.npy
│   ├── kos_binary.npy
│   ├── kos_vocab.npy
│   ├── nytimes_binary.npy
│   └── nytimes_vocab.npy
├── raw/                     # Raw datasets (auto-downloaded)
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
├── src/                     # Source code
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
├── experiment/              # Experiment scripts
│   ├── __init__.py
│   ├── experiment1.py
│   ├── experiment2.py
│   └── utils.py
├── scripts/                 # Data processing scripts
│   ├── convert.py
│   ├── filter_json.py
│   └── save_ground_truth.py
└── tests/                   # Test scripts
    └── test_pipeline.py
```