"""Utility to compute and save ground truth similarities to JSON file."""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src import cosine_similarity, jaccard_similarity, inner_product
from src.gpu_utils import GPUConfig, to_gpu, to_cpu, get_array_module


def calculate_ground_truth(
    X_dense: np.ndarray,
    similarity_score: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ground truth similarities for all pairs (GPU-accelerated).
    
    Args:
        X_dense: Dense binary matrix (n_samples, n_features)
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        
    Returns:
        Tuple of (ground_truth_values, pair_indices)
    """
    n_samples = X_dense.shape[0]
    pairs = list(combinations(range(n_samples), 2))
    n_pairs = len(pairs)
    
    print(f"Calculating Ground Truth ({similarity_score}) for {n_pairs} pairs...")
    
    use_gpu = GPUConfig.is_enabled()
    
    # For large datasets with GPU, use vectorized computation
    if use_gpu and n_pairs > 100:
        print(f"  Using GPU-accelerated batch computation...")
        X_gpu = to_gpu(X_dense)
        xp = get_array_module(X_gpu)
        
        # Compute in batches to avoid memory issues
        batch_size = min(10000, n_pairs)
        ground_truth = []
        
        for batch_start in tqdm(range(0, n_pairs, batch_size), desc="Computing ground truth", unit="batch"):
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            # Extract pairs as arrays
            indices_i = xp.array([p[0] for p in batch_pairs])
            indices_j = xp.array([p[1] for p in batch_pairs])
            
            # Get vectors for all pairs in batch
            vecs_i = X_gpu[indices_i]  # (batch_size, n_features)
            vecs_j = X_gpu[indices_j]  # (batch_size, n_features)
            
            if similarity_score == 'cosine_similarity':
                # Vectorized cosine similarity
                dot_products = xp.sum(vecs_i * vecs_j, axis=1)
                norms_i = xp.sqrt(xp.sum(vecs_i * vecs_i, axis=1))
                norms_j = xp.sqrt(xp.sum(vecs_j * vecs_j, axis=1))
                # Avoid division by zero
                denominators = norms_i * norms_j
                batch_sims = xp.where(denominators > 0, dot_products / denominators, 0.0)
            elif similarity_score == 'inner_product':
                # Vectorized inner product
                batch_sims = xp.sum(vecs_i * vecs_j, axis=1)
            elif similarity_score == 'jaccard_similarity':
                # Vectorized Jaccard similarity
                intersection = xp.sum(xp.minimum(vecs_i, vecs_j), axis=1)
                union = xp.sum(xp.maximum(vecs_i, vecs_j), axis=1)
                batch_sims = xp.where(union > 0, intersection / union, 0.0)
            else:
                raise ValueError(f"Unknown similarity score: {similarity_score}")
            
            # Transfer batch to CPU and extend results
            ground_truth.extend(to_cpu(batch_sims).tolist())
        
        return np.array(ground_truth), np.array(pairs)
    
    # CPU fallback or small datasets
    else:
        if use_gpu:
            print(f"  Using CPU (dataset too small for GPU batch optimization)...")
        ground_truth = []
        for i, j in tqdm(pairs, desc="Computing ground truth", unit="pair"):
            if similarity_score == 'cosine_similarity':
                val = cosine_similarity(X_dense[i], X_dense[j])
            elif similarity_score == 'inner_product':
                val = inner_product(X_dense[i], X_dense[j])
            elif similarity_score == 'jaccard_similarity':
                val = jaccard_similarity(X_dense[i], X_dense[j])
            else:
                raise ValueError(f"Unknown similarity score: {similarity_score}")
            ground_truth.append(val)
        
        return np.array(ground_truth), np.array(pairs)


def save_ground_truth_to_file(
    data_path: str,
    similarity_score: str,
    ground_truth: np.ndarray,
    pairs: np.ndarray,
    output_path: str = None
) -> str:
    """
    Save ground truth data to JSON file.
    
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
        dataset_name = Path(data_path).stem
        output_path = f"ground_truth_{dataset_name}_{similarity_score}.json"
    
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
    
    # Save to JSON
    print(f"\nSaving ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved {n_pairs} ground truth values")
    print(f"  Min similarity: {ground_truth.min():.6f}")
    print(f"  Max similarity: {ground_truth.max():.6f}")
    print(f"  Mean similarity: {ground_truth.mean():.6f}")
    
    return output_path


def calculate_and_save_ground_truth(
    data_path: str,
    similarity_score: str,
    output_path: str = None,
    use_gpu: bool = False
) -> str:
    """
    Calculate ground truth similarities and save to JSON.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        output_path: Output JSON file path (default: auto-generated)
        use_gpu: Whether to use GPU acceleration
        
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
    
    # Calculate ground truth
    ground_truth, pairs = calculate_ground_truth(X_dense, similarity_score)
    
    # Save to file
    return save_ground_truth_to_file(data_path, similarity_score, ground_truth, pairs, output_path)


def load_ground_truth(json_path: str):
    """
    Load ground truth from JSON file.
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and save ground truth similarities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        '--output', type=str, default=None,
        help='Output JSON file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration (requires CuPy and CUDA)'
    )
    
    parser.add_argument(
        '--load', type=str, default=None,
        help='Load and display ground truth from existing JSON file'
    )
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing ground truth
        load_ground_truth(args.load)
    else:
        # Calculate and save new ground truth
        calculate_and_save_ground_truth(
            args.data_path,
            args.similarity_score,
            args.output,
            args.use_gpu
        )


if __name__ == '__main__':
    main()
