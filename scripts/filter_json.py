"""Filter similarity scores from JSON ground truth files based on threshold."""
import json
import argparse
import numpy as np
from pathlib import Path


def filter_ground_truth_exp1(input_file, output_file, threshold, comparison='ge'):
    """
    Filter Experiment 1 ground truth JSON by similarity threshold.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        threshold: Similarity threshold value
        comparison: 'ge' (>=), 'gt' (>), 'le' (<=), 'lt' (<), 'eq' (==)
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    ground_truth = np.array(data['ground_truth'])
    pairs = np.array(data['pairs'])
    
    # Apply filter based on comparison
    if comparison == 'ge':
        mask = ground_truth >= threshold
    elif comparison == 'gt':
        mask = ground_truth > threshold
    elif comparison == 'le':
        mask = ground_truth <= threshold
    elif comparison == 'lt':
        mask = ground_truth < threshold
    elif comparison == 'eq':
        mask = np.isclose(ground_truth, threshold)
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    
    # Filter
    filtered_gt = ground_truth[mask]
    filtered_pairs = pairs[mask]
    
    # Update data
    data['ground_truth'] = filtered_gt.tolist()
    data['pairs'] = filtered_pairs.tolist()
    data['filter_applied'] = {
        'threshold': threshold,
        'comparison': comparison,
        'original_count': len(ground_truth),
        'filtered_count': len(filtered_gt)
    }
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Filtered {len(ground_truth)} -> {len(filtered_gt)} pairs")
    print(f"Saved to {output_file}")
    
    return data


def filter_ground_truth_exp2(input_file, output_file, threshold, comparison='ge'):
    """
    Filter Experiment 2 ground truth JSON by similarity threshold.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        threshold: Similarity threshold value
        comparison: 'ge' (>=), 'gt' (>), 'le' (<=), 'lt' (<), 'eq' (==)
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    similarities = np.array(data['similarities'])
    
    # Apply filter - count how many pairs above threshold
    if comparison == 'ge':
        mask = similarities >= threshold
    elif comparison == 'gt':
        mask = similarities > threshold
    elif comparison == 'le':
        mask = similarities <= threshold
    elif comparison == 'lt':
        mask = similarities < threshold
    elif comparison == 'eq':
        mask = np.isclose(similarities, threshold)
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    
    # Statistics
    n_query, n_train = similarities.shape
    total_pairs = n_query * n_train
    filtered_count = np.sum(mask)
    
    # Create filtered version (store as sparse representation)
    filtered_pairs = []
    for i in range(n_query):
        neighbors = []
        for j in range(n_train):
            if mask[i, j]:
                neighbors.append({
                    'train_idx': int(j),
                    'similarity': float(similarities[i, j])
                })
        filtered_pairs.append(neighbors)
    
    # Create output data
    output_data = {
        'data_path': data['data_path'],
        'similarity_score': data['similarity_score'],
        'n_train': data['n_train'],
        'n_query': data['n_query'],
        'train_indices': data['train_indices'],
        'query_indices': data['query_indices'],
        'filter_applied': {
            'threshold': threshold,
            'comparison': comparison,
            'total_pairs': int(total_pairs),
            'filtered_count': int(filtered_count),
            'percentage': float(filtered_count / total_pairs * 100)
        },
        'filtered_pairs': filtered_pairs
    }
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Total pairs: {total_pairs}")
    print(f"Filtered pairs: {filtered_count} ({filtered_count/total_pairs*100:.2f}%)")
    print(f"Average neighbors per query: {filtered_count/n_query:.2f}")
    print(f"Saved to {output_file}")
    
    return output_data


def analyze_json(input_file):
    """Analyze a ground truth JSON file."""
    print(f"\nAnalyzing {input_file}...")
    print("="*70)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Determine experiment type
    if 'pairs' in data:
        # Experiment 1
        print("Type: Experiment 1 (Accuracy)")
        ground_truth = np.array(data['ground_truth'])
        pairs = np.array(data['pairs'])
        
        print(f"Dataset: {data.get('data_path', 'N/A')}")
        print(f"Similarity: {data.get('similarity_score', 'N/A')}")
        print(f"Total pairs: {len(ground_truth)}")
        print(f"\nSimilarity Statistics:")
        print(f"  Min: {np.min(ground_truth):.6f}")
        print(f"  Max: {np.max(ground_truth):.6f}")
        print(f"  Mean: {np.mean(ground_truth):.6f}")
        print(f"  Median: {np.median(ground_truth):.6f}")
        print(f"  Std: {np.std(ground_truth):.6f}")
        
        # Distribution by threshold
        print(f"\nPairs above threshold:")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = np.sum(ground_truth >= thresh)
            pct = count / len(ground_truth) * 100
            print(f"  >= {thresh:.1f}: {count:6d} ({pct:5.2f}%)")
    
    elif 'similarities' in data:
        # Experiment 2
        print("Type: Experiment 2 (Retrieval)")
        similarities = np.array(data['similarities'])
        
        print(f"Dataset: {data.get('data_path', 'N/A')}")
        print(f"Similarity: {data.get('similarity_score', 'N/A')}")
        print(f"Query samples: {data['n_query']}")
        print(f"Train samples: {data['n_train']}")
        print(f"Total pairs: {similarities.size}")
        print(f"\nSimilarity Statistics:")
        print(f"  Min: {np.min(similarities):.6f}")
        print(f"  Max: {np.max(similarities):.6f}")
        print(f"  Mean: {np.mean(similarities):.6f}")
        print(f"  Median: {np.median(similarities):.6f}")
        print(f"  Std: {np.std(similarities):.6f}")
        
        # Distribution by threshold
        print(f"\nPairs above threshold:")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = np.sum(similarities >= thresh)
            pct = count / similarities.size * 100
            avg_neighbors = count / data['n_query']
            print(f"  >= {thresh:.1f}: {count:8d} ({pct:5.2f}%, avg {avg_neighbors:.1f} neighbors/query)")
    
    else:
        print("Unknown format")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Filter ground truth JSON files by similarity threshold'
    )
    
    parser.add_argument(
        'input_file',
        help='Input JSON file path'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: add _filtered suffix)'
    )
    
    parser.add_argument(
        '--threshold', type=float,
        help='Similarity threshold value'
    )
    
    parser.add_argument(
        '--comparison', default='ge',
        choices=['ge', 'gt', 'le', 'lt', 'eq'],
        help='Comparison operator (ge: >=, gt: >, le: <=, lt: <, eq: ==)'
    )
    
    parser.add_argument(
        '--analyze-only', action='store_true',
        help='Only analyze the file, do not filter'
    )
    
    args = parser.parse_args()
    
    # Analyze only
    if args.analyze_only:
        analyze_json(args.input_file)
        return
    
    # Filter requires threshold
    if args.threshold is None:
        print("Error: --threshold required for filtering")
        print("Use --analyze-only to just view statistics")
        return
    
    # Determine output file
    if args.output is None:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_filtered_t{args.threshold}{input_path.suffix}"
    else:
        output_file = args.output
    
    # Load and detect type
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Filter based on type
    if 'pairs' in data:
        filter_ground_truth_exp1(args.input_file, output_file, args.threshold, args.comparison)
    elif 'similarities' in data:
        filter_ground_truth_exp2(args.input_file, output_file, args.threshold, args.comparison)
    else:
        print("Error: Unknown JSON format")


if __name__ == '__main__':
    main()
