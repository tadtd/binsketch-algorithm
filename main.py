"""Main experiment dispatcher for BinSketch algorithm evaluation.

This script dispatches to different experiments:
- Experiment 1: Accuracy of Similarity Estimation (MSE-based)
- Experiment 2: Retrieval Performance (Precision/Recall/F1/Accuracy)
"""
import argparse
import sys

from src.gpu_utils import GPUConfig
from experiment import run_experiment1, run_experiment2


# Algorithm choices
ALGORITHM_CHOICES = ['BinSketch', 'BCS', 'MinHash', 'SimHash']

# Default compression lengths
DEFAULT_COMPRESSION_LENGTHS = [100, 500, 1000, 2000, 3000, 4000, 5000]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BinSketch Experiment Dispatcher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--experiment', type=str, required=True,
        choices=['1', '2', 'exp1', 'exp2'],
        help='Experiment to run: 1/exp1 for accuracy estimation, 2/exp2 for retrieval'
    )
    
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the .npy file containing the binary matrix'
    )
    
    parser.add_argument(
        '--algo', nargs='+', default=['BinSketch'],
        choices=ALGORITHM_CHOICES,
        help='Algorithms to run'
    )
    
    parser.add_argument(
        '--threshold', nargs='+', type=float, default=[0.9],
        help='Similarity thresholds to evaluate'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, default='cosine_similarity',
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric'
    )
    
    parser.add_argument(
        '--compression_lengths', nargs='+', type=int,
        default=None,
        help='Compression lengths to test (default: 100 500 1000 2000 3000 4000 5000)'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration (requires CuPy and CUDA)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Experiment 1 specific arguments
    parser.add_argument(
        '--eval_metric', type=str, default='minus_log_mse',
        choices=['mse', 'minus_log_mse'],
        help='[Exp1] Evaluation metric for comparing estimates to ground truth'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='[Exp1] Directory to save output plots'
    )
    
    parser.add_argument(
        '--ground_truth_path', type=str, default=None,
        help='[Exp1] Path to load/save ground truth JSON file'
    )
    
    # Experiment 2 specific arguments
    parser.add_argument(
        '--retrieval_metric', type=str, default='f1',
        choices=['precision', 'recall', 'f1', 'accuracy'],
        help='[Exp2] Retrieval metric to evaluate'
    )
    
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='[Exp2] Ratio of training data (0-1)'
    )
    
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='[Exp2] Path to save results JSON file'
    )
    
    return parser.parse_args()





def main():
    """Main entry point - dispatches to experiments."""
    args = parse_arguments()
    
    # Configure GPU
    if args.use_gpu:
        success = GPUConfig.enable_gpu()
        if not success:
            print("Warning: GPU not available, continuing with CPU")
    else:
        print("Running on CPU (use --use_gpu to enable GPU acceleration)")
    
    # Validate algorithms
    invalid_algos = [a for a in args.algo if a not in ALGORITHM_CHOICES]
    if invalid_algos:
        print(f"Error: Unknown algorithms: {invalid_algos}")
        print(f"Available algorithms: {ALGORITHM_CHOICES}")
        return
    
    # Set compression lengths
    compression_lengths = (
        args.compression_lengths
        if args.compression_lengths is not None
        else DEFAULT_COMPRESSION_LENGTHS
    )
    
    # Dispatch to appropriate experiment
    experiment = args.experiment.lower()
    
    if experiment in ['1', 'exp1']:
        print("="*60)
        print("Running Experiment 1: Accuracy of Similarity Estimation")
        print("="*60)
        
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
    
    elif experiment in ['2', 'exp2']:
        print("="*60)
        print("Running Experiment 2: Retrieval Performance Evaluation")
        print("="*60)
        
        run_experiment2(
            data_path=args.data_path,
            algorithms=args.algo,
            thresholds=args.threshold,
            similarity_score=args.similarity_score,
            retrieval_metric=args.retrieval_metric,
            compression_lengths=compression_lengths,
            train_ratio=args.train_ratio,
            seed=args.seed,
            output_path=args.output_path,
            ground_truth_path=args.ground_truth_path
        )
    
    else:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print("Use --experiment 1 or --experiment 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
