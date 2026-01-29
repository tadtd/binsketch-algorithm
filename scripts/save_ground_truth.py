"""Utility to compute and save ground truth similarities to JSON file.

Supports two modes:
1. Experiment 1: All pairs ground truth
2. Experiment 2: Train/query split with neighbors above threshold
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import Tuple
import numpy as np
from itertools import combinations
from tqdm import tqdm

from src import batch_inner_product, batch_cosine_similarity, batch_jaccard_similarity
from src.gpu_utils import to_gpu, to_cpu, get_array_module, GPUConfig


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
    output_path: str = None
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
    
    # Generate output path if not provided
    if output_path is None:
        dataset_name = Path(data_path).stem.replace('_binary', '')
        output_path = f"experiment/ground_truth/ground_truth_{dataset_name}_{similarity_score}.json"
    
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    print(f"\nSaving ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved {n_pairs} ground truth values")
    print(f"  Min similarity: {ground_truth.min():.6f}")
    print(f"  Max similarity: {ground_truth.max():.6f}")
    print(f"  Mean similarity: {ground_truth.mean():.6f}")
    
    return output_path


def calculate_and_save_experiment1_ground_truth(
    data_path: str,
    similarity_score: str,
    output_path: str = None,
    use_gpu: bool = False
) -> str:
    """Calculate Experiment 1 ground truth similarities and save to JSON.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        output_path: Output JSON file path (default: auto-generated)
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Path to saved JSON file
    """
    # Enable GPU if requested
    if use_gpu:
        GPUConfig.enable_gpu()
    
    # Load data
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    print(f"  Matrix Shape: {X_dense.shape}")
    print(f"  Data type: {X_dense.dtype}")
    print(f"  Value range: [{X_dense.min()}, {X_dense.max()}]")
    
    # Ensure data is binary (0 or 1)
    if X_dense.min() < 0 or X_dense.max() > 1 or not np.all(np.isin(X_dense, [0, 1])):
        print(f"⚠ Warning: Data is not binary! Converting to binary (0/1)...")
        unique_vals = np.unique(X_dense)
        print(f"  Unique values in data: {unique_vals[:10]}...")  # Show first 10
        # Convert to binary: any non-zero value becomes 1
        X_dense = (X_dense != 0).astype(np.float32)
        print(f"  After conversion - range: [{X_dense.min()}, {X_dense.max()}]")
    else:
        # Ensure float32 for GPU efficiency
        X_dense = X_dense.astype(np.float32)
    
    # Calculate ground truth
    ground_truth, pairs = calculate_experiment1_ground_truth(X_dense, similarity_score, use_gpu)
    
    # Save to file
    return save_experiment1_ground_truth(data_path, similarity_score, ground_truth, pairs, output_path)


def load_experiment1_ground_truth(json_path: str):
    """Load Experiment 1 ground truth from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        dict with keys: data_path, similarity_score, n_samples, n_features, n_pairs, pairs, ground_truth
    """
    print(f"Loading ground truth from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    data['pairs'] = np.array(data['pairs'])
    data['ground_truth'] = np.array(data['ground_truth'])
    
    print(f"✓ Loaded {data['n_pairs']} ground truth values")
    print(f"  Dataset: {data['data_path']}")
    print(f"  Similarity: {data['similarity_score']}")
    print(f"  Shape: ({data['n_samples']}, {data['n_features']})")
    
    return data


def calculate_experiment2_ground_truth(
    X_train: np.ndarray,
    X_query: np.ndarray,
    similarity_score: str,
    use_gpu: bool = False,
    chunk_size: int = 100
) -> np.ndarray:
    """
    Calculate ground truth similarities for Experiment 2 (train/query split).
    Computes similarity matrix between query and training vectors.
    Uses vectorized GPU operations with chunked processing for progress tracking.
    
    Args:
        X_train: Training vectors (n_train, n_features)
        X_query: Query vectors (n_query, n_features)
        similarity_score: Similarity metric
        use_gpu: Whether to use GPU for computations
        chunk_size: Number of query rows to process per chunk
        
    Returns:
        Similarity matrix (n_query, n_train)
    """
    # Transfer to GPU if requested
    if use_gpu:
        print("Transferring data to GPU...")
        X_train = to_gpu(X_train)
        X_query = to_gpu(X_query)
        xp = get_array_module(X_train)
        print(f"✓ Data transferred to GPU (using {xp.__name__})")
    else:
        xp = get_array_module(X_train)
    
    # Ensure proper dtype to avoid overflow (CRITICAL for inner product!)
    X_train = X_train.astype(xp.float32)
    X_query = X_query.astype(xp.float32)
    
    device = "GPU" if use_gpu else "CPU"
    print(f"Calculating Experiment 2 Ground Truth ({similarity_score})...")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Query samples: {X_query.shape[0]}")
    print(f"  Using GPU-accelerated batch computation...")
    
    n_query = X_query.shape[0]
    n_train = X_train.shape[0]
    
    # Compute similarity matrix in chunks with progress bar
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    similarities = xp.zeros((n_query, n_train), dtype=xp.float32)
    
    with tqdm(total=n_chunks, desc="Computing similarities", unit="batch",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_query)
            query_chunk = X_query[start_idx:end_idx]
            
            # Compute similarity for this chunk against all training samples
            if similarity_score == 'inner_product':
                similarities[start_idx:end_idx] = batch_inner_product(query_chunk, X_train)
            elif similarity_score == 'cosine_similarity':
                similarities[start_idx:end_idx] = batch_cosine_similarity(query_chunk, X_train)
            elif similarity_score == 'jaccard_similarity':
                similarities[start_idx:end_idx] = batch_jaccard_similarity(query_chunk, X_train)
            else:
                raise ValueError(f"Unknown similarity score: {similarity_score}")
            
            pbar.update(1)
    
    # Transfer back to CPU for JSON serialization
    similarities = to_cpu(similarities)
    
    print(f"\n  Min similarity: {similarities.min():.6f}")
    print(f"  Max similarity: {similarities.max():.6f}")
    print(f"  Mean similarity: {similarities.mean():.6f}")
    
    return similarities


def save_experiment2_ground_truth(
    data_path: str,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    similarity_score: str,
    similarities: np.ndarray,
    output_path: str = None
) -> str:
    """
    Save Experiment 2 ground truth to JSON file.
    
    Args:
        data_path: Path to original data file
        train_indices: Training set indices
        query_indices: Query set indices
        similarity_score: Similarity metric
        similarities: Similarity matrix (n_query, n_train)
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    # Load data for metadata
    X_dense = np.load(data_path)
    n_samples, n_features = X_dense.shape
    
    # Generate output path if not provided
    if output_path is None:
        dataset_name = Path(data_path).stem.replace('_binary', '')
        output_path = f"experiment/ground_truth/ground_truth_exp2_{dataset_name}_{similarity_score}.json"
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON
    data_to_save = {
        "experiment": 2,
        "data_path": data_path,
        "similarity_score": similarity_score,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_train": len(train_indices),
        "n_query": len(query_indices),
        "train_indices": train_indices.tolist(),
        "query_indices": query_indices.tolist(),
        "similarities": similarities.tolist()
    }
    
    # Save to JSON
    print(f"\nSaving ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved similarity matrix ({similarities.shape[0]}, {similarities.shape[1]})")
    
    return output_path


def calculate_and_save_experiment2_ground_truth(
    data_path: str,
    train_ratio: float,
    similarity_score: str,
    seed: int = 42,
    output_path: str = None,
    use_gpu: bool = False
) -> str:
    """
    Calculate and save Experiment 2 ground truth.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        train_ratio: Ratio for training set
        similarity_score: Similarity metric
        seed: Random seed
        output_path: Output JSON file path (default: auto-generated)
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Path to saved JSON file
    """
    # Enable GPU if requested
    if use_gpu:
        GPUConfig.enable_gpu()
    
    # Load data
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    print(f"  Matrix Shape: {X_dense.shape}")
    print(f"  Data type: {X_dense.dtype}")
    print(f"  Value range: [{X_dense.min()}, {X_dense.max()}]")
    
    # Ensure data is binary (0 or 1)
    if X_dense.min() < 0 or X_dense.max() > 1 or not np.all(np.isin(X_dense, [0, 1])):
        print(f"⚠ Warning: Data is not binary! Converting to binary (0/1)...")
        unique_vals = np.unique(X_dense)
        print(f"  Unique values in data: {unique_vals[:10]}...")  # Show first 10
        # Convert to binary: any non-zero value becomes 1
        X_dense = (X_dense != 0).astype(np.float32)
        print(f"  After conversion - range: [{X_dense.min()}, {X_dense.max()}]")
    else:
        # Ensure float32 for GPU efficiency
        X_dense = X_dense.astype(np.float32)
    
    # Split dataset
    np.random.seed(seed)
    n_samples = X_dense.shape[0]
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    query_indices = indices[n_train:]
    
    X_train = X_dense[train_indices]
    X_query = X_dense[query_indices]
    
    print(f"Dataset split: {len(train_indices)} training, {len(query_indices)} query")
    
    # Calculate ground truth similarities
    similarities = calculate_experiment2_ground_truth(
        X_train, X_query, similarity_score, use_gpu
    )
    
    # Save to file
    return save_experiment2_ground_truth(
        data_path, train_indices, query_indices,
        similarity_score, similarities, output_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and save ground truth similarities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--experiment', type=int, choices=[1, 2], default=1,
        help='Experiment type: 1 (all pairs) or 2 (train/query split)'
    )
    
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the .npy file containing the binary matrix'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, required=True,
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric for ground truth calculation'
    )
    
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='Training set ratio (for experiment 2)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (for experiment 2)'
    )
    
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--load', type=str, default=None,
        help='Load and display ground truth from existing JSON file'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Use GPU for computations if available'
    )
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing ground truth
        load_experiment1_ground_truth(args.load)
    elif args.experiment == 1:
        # Calculate and save Experiment 1 ground truth
        calculate_and_save_experiment1_ground_truth(
            args.data_path,
            args.similarity_score,
            args.output,
            args.use_gpu
        )
    elif args.experiment == 2:
        # Calculate and save Experiment 2 ground truth
        calculate_and_save_experiment2_ground_truth(
            args.data_path,
            args.train_ratio,
            args.similarity_score,
            args.seed,
            args.output,
            args.use_gpu
        )


if __name__ == '__main__':
    main()
