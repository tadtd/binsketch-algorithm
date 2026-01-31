"""Experiment 1: Accuracy of Similarity Estimation.

This experiment evaluates how accurately different sketching algorithms
can estimate pairwise similarities compared to ground truth values.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import List, Optional, Tuple
import numpy as np
from itertools import combinations
from tqdm import tqdm

from src import BinSketch, BinaryCompressionSchema, MinHash, SimHash
from src import batch_inner_product, batch_cosine_similarity, batch_jaccard_similarity
from src.gpu_utils import GPUConfig, to_gpu, to_cpu, get_array_module

# Import from local utils (works both as module and standalone)

from experiment.utils import (
    load_data, filter_pairs_by_threshold, get_estimator, get_evaluation_metric,
    plot_experiment1_results
)


# Algorithm mapping
ALGO_MAP = {
    'BinSketch': BinSketch,
    'BCS': BinaryCompressionSchema,
    'MinHash': MinHash,
    'SimHash': SimHash
}

# Default compression lengths
DEFAULT_COMPRESSION_LENGTHS = [100, 500, 1000, 2000, 3000, 4000, 5000]


# ============================================================================
# Ground Truth Utility Functions
# ============================================================================

def calculate_experiment1_ground_truth(
    X_dense: np.ndarray,
    similarity_score: str,
    use_gpu: bool = False,
    chunk_size: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate ground truth similarities for all pairs (Experiment 1).
    Uses vectorized GPU operations with chunked processing for progress tracking.
    
    Args:
        X_dense: Dense binary matrix (n_samples, n_features)
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        use_gpu: Whether to use GPU for computations
        chunk_size: Number of rows to process per chunk (for progress tracking)
        
    Returns:
        Tuple of (ground_truth_values, pair_indices)
    """
    # Transfer to GPU if requested
    if use_gpu:
        print("Transferring data to GPU...")
        X_dense = to_gpu(X_dense)
        xp = get_array_module(X_dense)
        print(f"✓ Data transferred to GPU (using {xp.__name__})")
    else:
        xp = get_array_module(X_dense)
    
    # Ensure proper dtype to avoid overflow (CRITICAL for inner product!)
    X_dense = X_dense.astype(xp.float32)
    
    n_samples = X_dense.shape[0]
    pairs = list(combinations(range(n_samples), 2))
    n_pairs = len(pairs)
    
    device = "GPU" if use_gpu else "CPU"
    print(f"Calculating Ground Truth ({similarity_score}) for {n_pairs} pairs...")
    if use_gpu:
        print(f"  Using GPU-accelerated batch computation...")
    
    # Compute similarity matrix in chunks with progress bar
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    sim_matrix = xp.zeros((n_samples, n_samples), dtype=xp.float32)
    
    with tqdm(total=n_chunks, desc="Computing ground truth", unit="batch", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_samples)
            chunk = X_dense[start_idx:end_idx]
            
            # Compute similarity for this chunk against all samples
            if similarity_score == 'inner_product':
                sim_matrix[start_idx:end_idx] = batch_inner_product(chunk, X_dense)
            elif similarity_score == 'cosine_similarity':
                sim_matrix[start_idx:end_idx] = batch_cosine_similarity(chunk, X_dense)
            elif similarity_score == 'jaccard_similarity':
                sim_matrix[start_idx:end_idx] = batch_jaccard_similarity(chunk, X_dense)
            else:
                raise ValueError(f"Unknown similarity score: {similarity_score}")
            
            pbar.update(1)
    
    # Extract upper triangle (all pairs)
    print("Extracting pair similarities...")
    pairs_array = xp.array(pairs)
    ground_truth = sim_matrix[pairs_array[:, 0], pairs_array[:, 1]]
    
    # Transfer to CPU
    ground_truth = to_cpu(ground_truth)
    
    return ground_truth, np.array(pairs)


def save_experiment1_ground_truth(
    data_path: str,
    similarity_score: str,
    ground_truth: np.ndarray,
    pairs: np.ndarray,
    output_path: Optional[str] = None
) -> str:
    """Save Experiment 1 ground truth data to JSON file.
    
    Args:
        data_path: Path to original data file
        similarity_score: Similarity metric used
        ground_truth: Ground truth values array
        pairs: Pairs array
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    # Load data for metadata
    X_dense = np.load(data_path)
    n_samples, n_features = X_dense.shape
    n_pairs = len(pairs)
    
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # Generate output path if not provided
    if output_path is None:
        filename = f"ground_truth_exp1_{dataset_name}_{similarity_score}.json"
    else:
        # Replace template placeholders if present
        filename = output_path.replace('{DATASET}', dataset_name).replace('{SIMILARITY_SCORE}', similarity_score).replace('{SIMILARITY}', similarity_score)
    
    # Always place in experiment/ground_truth/ directory
    filepath = f"experiment/ground_truth/{filename}"
    
    # Prepare data for JSON
    data_to_save = {
        "data_path": data_path,
        "similarity_score": similarity_score,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_pairs": int(n_pairs),
        "pairs": pairs.tolist(),
        "ground_truth": ground_truth.tolist()
    }
    
    # Create parent directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    print(f"\nSaving ground truth to {filepath}...")
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved {n_pairs} ground truth values")
    print(f"  Min similarity: {ground_truth.min():.6f}")
    print(f"  Max similarity: {ground_truth.max():.6f}")
    print(f"  Mean similarity: {ground_truth.mean():.6f}")
    
    return filepath


def load_experiment1_ground_truth(
    data_path: str,
    similarity_score: str,
    ground_truth_path: Optional[str] = None
) -> Optional[dict]:
    """Load Experiment 1 ground truth from JSON file if it exists.
    
    Args:
        data_path: Original data path
        similarity_score: Similarity metric name
        ground_truth_path: Path to ground truth JSON file (default: auto-generated)
        
    Returns:
        dict with keys: data_path, similarity_score, n_samples, n_features, n_pairs, pairs, ground_truth
        or None if file doesn't exist
    """
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # Generate filepath if not provided
    if ground_truth_path is None:
        filename = f"ground_truth_exp1_{dataset_name}_{similarity_score}.json"
    else:
        # Replace template placeholders if present
        filename = ground_truth_path.replace('{DATASET}', dataset_name).replace('{SIMILARITY_SCORE}', similarity_score).replace('{SIMILARITY}', similarity_score)
    
    # Always place in experiment/ground_truth/ directory
    filepath = f"experiment/ground_truth/{filename}"
    
    if not Path(filepath).exists():
        return None
    
    print(f"Loading ground truth from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    data['pairs'] = np.array(data['pairs'])
    data['ground_truth'] = np.array(data['ground_truth'])
    
    print(f"✓ Loaded {data['n_pairs']} ground truth values")
    print(f"  Dataset: {data['data_path']}")
    print(f"  Similarity: {data['similarity_score']}")
    print(f"  Shape: ({data['n_samples']}, {data['n_features']})")
    
    return data


# ============================================================================
# Main Experiment Function
# ============================================================================

def run_experiment1(
    data_path: str,
    algorithms: List[str],
    thresholds: List[float],
    similarity_score: str,
    eval_metric: str,
    compression_lengths: List[int],
    output_dir: str,
    seed: int = 42,
    ground_truth_path: Optional[str] = None
) -> None:
    """
    Run Experiment 1: Accuracy of Similarity Estimation.
    
    This experiment compares estimated similarities from compressed representations
    against ground truth similarities from the original data.
    
    OPTIMIZED: Computes sketches once per (algorithm, k), then evaluates all thresholds.
    
    Args:
        data_path: Path to data file
        algorithms: List of algorithm names to test
        thresholds: List of similarity thresholds
        similarity_score: Similarity metric name ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        eval_metric: Evaluation metric name ('mse', 'minus_log_mse')
        compression_lengths: List of compression lengths to test
        output_dir: Output directory for results
        seed: Random seed
        ground_truth_path: Optional path to ground truth JSON file
    """
    from experiment.utils import get_estimator, get_evaluation_metric
    
    X_dense, X_csr = load_data(data_path)
    
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # =========================================================================
    # Step 1: Load or compute ground truth ONCE
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Loading/Computing Ground Truth")
    print(f"{'='*60}")
    
    gt_data = load_experiment1_ground_truth(data_path, similarity_score, ground_truth_path)
    
    if gt_data is not None:
        ground_truth = gt_data['ground_truth']
        pair_indices = gt_data['pairs']
        print(f"Using cached ground truth")
    else:
        print(f"Calculating ground truth (this may take a while)...")
        ground_truth, pair_indices = calculate_experiment1_ground_truth(
            X_dense, similarity_score, use_gpu=GPUConfig.is_enabled()
        )
        save_experiment1_ground_truth(
            data_path, similarity_score, ground_truth, pair_indices,
            output_path=ground_truth_path
        )
    
    # =========================================================================
    # Step 2: Precompute filtered pairs for all thresholds
    # =========================================================================
    print(f"\nPrecomputing filtered pairs for {len(thresholds)} thresholds...")
    filtered_data_per_threshold = {}
    for threshold in thresholds:
        gt_filtered, pairs_filtered = filter_pairs_by_threshold(
            ground_truth, pair_indices, threshold
        )
        if gt_filtered is not None:
            filtered_data_per_threshold[threshold] = {
                'gt': gt_filtered,
                'pairs': pairs_filtered
            }
            print(f"  Threshold {threshold}: {len(gt_filtered)} pairs")
        else:
            print(f"  Threshold {threshold}: No pairs found")
    
    # =========================================================================
    # Step 3: Find the union of all pairs needed (lowest threshold covers all)
    # =========================================================================
    # The lowest threshold will have the most pairs, which is a superset of higher thresholds
    min_threshold = min(filtered_data_per_threshold.keys())
    all_needed_pairs = filtered_data_per_threshold[min_threshold]['pairs']
    print(f"\nTotal unique pairs to estimate: {len(all_needed_pairs)} (from threshold {min_threshold})")
    
    # Create mapping from pair to index for fast lookup
    pair_to_idx = {tuple(p): idx for idx, p in enumerate(all_needed_pairs)}
    
    # Initialize results structure
    # Structure: {threshold: {algo_name: [scores_per_k]}}
    all_results = {t: {} for t in thresholds if t in filtered_data_per_threshold}
    
    # Get evaluation function
    eval_func = get_evaluation_metric(eval_metric)
    
    # =========================================================================
    # Step 4: For each algorithm and k, compute sketches ONCE, 
    #         estimate only needed pairs, then evaluate each threshold
    # =========================================================================
    for algo_name in algorithms:
        if algo_name not in ALGO_MAP:
            print(f"\nUnknown algorithm: {algo_name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Algorithm: {algo_name}")
        print(f"{'='*60}")
        
        # Initialize results for this algorithm across all thresholds
        for threshold in all_results:
            all_results[threshold][algo_name] = []
        
        for k in compression_lengths:
            print(f"\n  Compression length k={k}")
            
            # Create model and compute sketches ONCE
            model = ALGO_MAP[algo_name](seed=seed)
            
            print(f"    Computing sketches...")
            try:
                X_sketch = model.mapping(X_csr, k=k)
            except Exception as e:
                print(f"    Error: Failed to compress with k={k}: {e}")
                # Append NaN for all thresholds
                for threshold in all_results:
                    all_results[threshold][algo_name].append(float('nan'))
                continue
            
            # Estimate similarities for only the needed pairs ONCE
            print(f"    Estimating similarities for {len(all_needed_pairs)} pairs...")
            estimator = get_estimator(model, algo_name, similarity_score)
            
            all_estimates = np.zeros(len(all_needed_pairs))
            for idx, (i, j) in enumerate(tqdm(all_needed_pairs, desc="Estimating", unit="pair", leave=False)):
                all_estimates[idx] = estimator(X_sketch[i], X_sketch[j])
            
            # Now evaluate each threshold using pre-computed estimates
            for threshold in all_results:
                if threshold not in filtered_data_per_threshold:
                    continue
                
                gt_filtered = filtered_data_per_threshold[threshold]['gt']
                pairs_filtered = filtered_data_per_threshold[threshold]['pairs']
                
                # Extract estimates for filtered pairs using the mapping
                est_filtered = np.array([
                    all_estimates[pair_to_idx[tuple(p)]] 
                    for p in pairs_filtered
                ])
                
                # Compute evaluation metric
                score = eval_func(est_filtered, gt_filtered)
                
                if eval_metric == 'minus_log_mse' and np.isinf(score):
                    score = -np.log(1e-10)
                
                print(f"    Threshold {threshold}: {eval_metric} = {score:.4f}")
                all_results[threshold][algo_name].append(score)
    
    # =========================================================================
    # Step 5: Create combined multi-threshold plot
    # =========================================================================
    if all_results:
        # Filter out thresholds with no results
        valid_results = {t: r for t, r in all_results.items() if r}
        if valid_results:
            plot_experiment1_results(
                valid_results, compression_lengths, similarity_score, 
                eval_metric, dataset_name, "results/experiment1"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 1: Accuracy of Similarity Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the .npy file containing the binary matrix'
    )
    
    parser.add_argument(
        '--algo', nargs='+', default=['BinSketch'],
        choices=list(ALGO_MAP.keys()),
        help='Algorithms to run'
    )
    
    parser.add_argument(
        '--threshold', nargs='+', type=float, default=[0.9],
        help='Similarity thresholds to evaluate'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration (requires CuPy and CUDA)'
    )
    
    parser.add_argument(
        '--eval_metric', type=str, default='minus_log_mse',
        choices=['mse', 'minus_log_mse'],
        help='Evaluation metric for comparing estimates to ground truth'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, default='cosine_similarity',
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric for ground truth calculation and estimation'
    )
    
    parser.add_argument(
        '--compression_lengths', nargs='+', type=int,
        default=None,
        help='Compression lengths to test (default: 100 500 1000 2000 3000 4000 5000)'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory to save output plots'
    )
    
    parser.add_argument(
        '--ground_truth_path', type=str, default=None,
        help='Path to load/save ground truth JSON file'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.use_gpu:
        success = GPUConfig.enable_gpu()
        if not success:
            print("Warning: GPU not available, continuing with CPU")
    else:
        print("Running on CPU (use --use_gpu to enable GPU acceleration)")
    
    invalid_algos = [a for a in args.algo if a not in ALGO_MAP]
    if invalid_algos:
        print(f"Error: Unknown algorithms: {invalid_algos}")
        print(f"Available algorithms: {list(ALGO_MAP.keys())}")
        return
    
    compression_lengths = (
        args.compression_lengths
        if args.compression_lengths is not None
        else DEFAULT_COMPRESSION_LENGTHS
    )
    
    run_experiment1(
        data_path=args.data_path,
        algorithms=args.algo,
        thresholds=args.threshold,
        similarity_score=args.similarity_score,
        eval_metric=args.eval_metric,
        compression_lengths=compression_lengths,
        output_dir=args.output_dir,
        seed=args.seed,
        ground_truth_path=args.ground_truth_path
    )


if __name__ == '__main__':
    main()
