"""
Comprehensive testing pipeline for multiple sketching algorithms.
Tests: BinSketch, BCS, MinHash, SimHash on the same datasets.

Usage:
    python test_pipeline.py                    # Test all algorithms
    python test_pipeline.py --algo BinSketch   # Test only BinSketch
    python test_pipeline.py --algo BCS         # Test only BCS
    python test_pipeline.py --algo BinSketch BCS  # Test multiple specific algorithms
"""

import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random
from src.models.binsketch import BinSketch
from src.models.bcs import BinarySchemaCompression
from src.models.minhash import MinHash
from src.models.simhash import SimHash
from src.similarity_scores import inner_product, hamming_distance, jaccard_similarity, cosine_similarity
from src.metric import mse, precision, recall, f1_score
import time
import argparse
import json
import csv
from datetime import datetime
import os


def generate_random_sparse_matrix(n_rows=100, n_cols=100, density=0.1, seed=42):
    """Generate a random sparse binary matrix."""
    rng = np.random.RandomState(seed)
    X = sparse_random(n_rows, n_cols, density=density, format='csr', random_state=rng)
    X.data = np.ones_like(X.data)
    return X


class AlgorithmTester:
    """Test framework for sketching algorithms."""
    
    def __init__(self, seed=42):
        self.seed = seed
        self.algorithms = {
            'BinSketch': BinSketch(seed=seed),
            'BCS': BinarySchemaCompression(seed=seed),
            'MinHash': MinHash(seed=seed),
            'SimHash': SimHash(seed=seed)
        }
        self.results = {}
    
    def test_inner_product(self, X, k=50, num_pairs=20, algorithm_name='BinSketch'):
        """Test an algorithm for inner product estimation."""
        model = self.algorithms[algorithm_name]
        
        # Create sketches
        start_time = time.time()
        X_sketch = model.mapping(X, k)
        sketch_time = time.time() - start_time
        
        # Test on random pairs
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) 
                 for _ in range(num_pairs)]
        
        ground_truth = []
        estimates = []
        
        start_time = time.time()
        for i, j in pairs:
            true_ip = inner_product(X[i], X[j].T)
            ground_truth.append(true_ip)
            
            est_ip = model.estimate_inner_product(X_sketch[i], X_sketch[j])
            estimates.append(est_ip)
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        ground_truth = np.array(ground_truth)
        estimates = np.array(estimates)
        
        mse_val = mse(ground_truth, estimates)
        mae = np.mean(np.abs(ground_truth - estimates))
        max_error = np.max(np.abs(ground_truth - estimates))
        relative_error = np.mean(np.abs(ground_truth - estimates) / (np.abs(ground_truth) + 1e-10))
        
        return {
            'algorithm': algorithm_name,
            'metric': 'inner_product',
            'sketch_time': sketch_time,
            'test_time': test_time,
            'mse': mse_val,
            'mae': mae,
            'max_error': max_error,
            'relative_error': relative_error,
            'total_time': sketch_time + test_time,
            'compression_ratio': X.shape[1] / k,
            'matrix_rows': X.shape[0],
            'matrix_cols': X.shape[1],
            'sketch_size': k
        }
    
    def test_jaccard_similarity(self, X, k=50, num_pairs=20, algorithm_name='MinHash'):
        """Test an algorithm for Jaccard similarity estimation."""
        model = self.algorithms[algorithm_name]
        
        # Create sketches
        start_time = time.time()
        X_sketch = model.mapping(X, k)
        sketch_time = time.time() - start_time
        
        # Test on random pairs
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) 
                 for _ in range(num_pairs)]
        
        ground_truth = []
        estimates = []
        
        start_time = time.time()
        for i, j in pairs:
            true_jaccard = jaccard_similarity(X[i], X[j])
            ground_truth.append(true_jaccard)
            
            est_jaccard = model.estimate_jaccard_similarity(X_sketch[i], X_sketch[j])
            estimates.append(est_jaccard)
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        ground_truth = np.array(ground_truth)
        estimates = np.array(estimates)
        
        mse_val = mse(ground_truth, estimates)
        mae = np.mean(np.abs(ground_truth - estimates))
        max_error = np.max(np.abs(ground_truth - estimates))
        relative_error = np.mean(np.abs(ground_truth - estimates) / (np.abs(ground_truth) + 1e-10))
        
        return {
            'algorithm': algorithm_name,
            'metric': 'jaccard',
            'sketch_time': sketch_time,
            'test_time': test_time,
            'mse': mse_val,
            'mae': mae,
            'max_error': max_error,
            'relative_error': relative_error,
            'total_time': sketch_time + test_time,
            'compression_ratio': X.shape[1] / k,
            'matrix_rows': X.shape[0],
            'matrix_cols': X.shape[1],
            'sketch_size': k
        }
    
    def test_hamming_distance(self, X, k=50, num_pairs=20, algorithm_name='SimHash'):
        """Test an algorithm for Hamming distance estimation."""
        model = self.algorithms[algorithm_name]
        
        # Create sketches
        start_time = time.time()
        X_sketch = model.mapping(X, k)
        sketch_time = time.time() - start_time
        
        # Test on random pairs
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) 
                 for _ in range(num_pairs)]
        
        ground_truth = []
        estimates = []
        
        start_time = time.time()
        for i, j in pairs:
            true_hamming = hamming_distance(X[i], X[j])
            ground_truth.append(true_hamming)
            
            est_hamming = model.estimate_hamming_distance(X_sketch[i], X_sketch[j])
            estimates.append(est_hamming)
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        ground_truth = np.array(ground_truth)
        estimates = np.array(estimates)
        
        mse_val = mse(ground_truth, estimates)
        mae = np.mean(np.abs(ground_truth - estimates))
        max_error = np.max(np.abs(ground_truth - estimates))
        relative_error = np.mean(np.abs(ground_truth - estimates) / (np.abs(ground_truth) + 1e-10))
        
        return {
            'algorithm': algorithm_name,
            'metric': 'hamming',
            'sketch_time': sketch_time,
            'test_time': test_time,
            'mse': mse_val,
            'mae': mae,
            'max_error': max_error,
            'relative_error': relative_error,
            'total_time': sketch_time + test_time,
            'compression_ratio': X.shape[1] / k,
            'matrix_rows': X.shape[0],
            'matrix_cols': X.shape[1],
            'sketch_size': k
        }
    
    def test_cosine_similarity(self, X, k=50, num_pairs=20, algorithm_name='SimHash'):
        """Test an algorithm for Cosine similarity estimation."""
        model = self.algorithms[algorithm_name]
        
        # Create sketches
        start_time = time.time()
        X_sketch = model.mapping(X, k)
        sketch_time = time.time() - start_time
        
        # Test on random pairs
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) 
                 for _ in range(num_pairs)]
        
        ground_truth = []
        estimates = []
        
        start_time = time.time()
        for i, j in pairs:
            true_cosine = cosine_similarity(X[i], X[j].T)
            ground_truth.append(true_cosine)
            
            if hasattr(model, 'estimate_cosine_similarity'):
                est_cosine = model.estimate_cosine_similarity(X_sketch[i], X_sketch[j])
            else:
                raise NotImplementedError(f"{algorithm_name} does not support estimate_cosine_similarity")
            estimates.append(est_cosine)
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        ground_truth = np.array(ground_truth)
        estimates = np.array(estimates)
        
        mse_val = mse(ground_truth, estimates)
        mae = np.mean(np.abs(ground_truth - estimates))
        max_error = np.max(np.abs(ground_truth - estimates))
        relative_error = np.mean(np.abs(ground_truth - estimates) / (np.abs(ground_truth) + 1e-10))
        
        return {
            'algorithm': algorithm_name,
            'metric': 'cosine',
            'sketch_time': sketch_time,
            'test_time': test_time,
            'mse': mse_val,
            'mae': mae,
            'max_error': max_error,
            'relative_error': relative_error,
            'total_time': sketch_time + test_time,
            'compression_ratio': X.shape[1] / k,
            'matrix_rows': X.shape[0],
            'matrix_cols': X.shape[1],
            'sketch_size': k
        }
    
    def run_pipeline(self, X, k=50, num_pairs=20, algorithms=None, metrics=None):
        """Run comprehensive test pipeline on specified algorithms.
        
        Parameters:
        -----------
        X : sparse matrix
            Input data matrix
        k : int
            Sketch size
        num_pairs : int
            Number of test pairs
        algorithms : list of str or None
            List of algorithm names to test. If None, test all available algorithms.
            Options: 'BinSketch', 'BCS', 'MinHash', 'SimHash'
        metrics : list of str or None
            List of metrics to test. If None or ['all'], test all metrics.
            Options: 'inner_product', 'jaccard', 'hamming', 'cosine', 'all'
        """
        results = []
        
        # Determine which metrics to test
        if metrics is None or 'all' in metrics:
            test_metrics = ['inner_product', 'jaccard', 'hamming', 'cosine']
        else:
            test_metrics = metrics.copy()
            
        # If both cosine and inner_product are requested, prioritize cosine (remove inner_product)
        # unless user explicitly asked for inner_product separately? 
        # The user requested: "If I added cosine metric... remove inner_product"
        # Determine which algorithms to test
        if algorithms is None:
            test_inner_product = ['BinSketch', 'BCS']
            test_jaccard = ['MinHash', 'BinSketch'] # BinSketch supports Jaccard
            test_hamming = ['SimHash', 'BinSketch'] # BinSketch supports Hamming
            test_cosine = ['BinSketch', 'SimHash', 'BCS']
        else:
            # Filter algorithms by type
            # BinSketch is versatile so it appears in almost all lists if selected
            test_inner_product = [a for a in algorithms if a in ['BinSketch', 'BCS']]
            test_jaccard = [a for a in algorithms if a in ['MinHash', 'BinSketch']]
            test_hamming = [a for a in algorithms if a in ['SimHash', 'BinSketch']]
            test_cosine = [a for a in algorithms if a in ['BinSketch', 'SimHash', 'BCS']]
        
        # Test algorithms for inner product
        if test_inner_product and 'inner_product' in test_metrics:
            print("\n" + "="*80)
            print("Testing Inner Product Estimation")
            print("="*80)
            for algo in test_inner_product:
                print(f"\nTesting {algo}...")
                try:
                    result = self.test_inner_product(X, k, num_pairs, algo)
                    results.append(result)
                    print(f"[OK] MSE: {result['mse']:<10.4f} MAE: {result['mae']:<10.4f}")
                except NotImplementedError:
                    print(f"[!] {algo} not fully implemented yet, skipping...")
                except Exception as e:
                    print(f"[X] Error testing {algo}: {e}")
        
        # Test MinHash for Jaccard similarity
        if test_jaccard and 'jaccard' in test_metrics:
            print("\n" + "="*80)
            print("Testing Jaccard Similarity Estimation")
            print("="*80)
            for algo in test_jaccard:
                print(f"\nTesting {algo}...")
                try:
                    result = self.test_jaccard_similarity(X, k, num_pairs, algo)
                    results.append(result)
                    print(f"[OK] MSE: {result['mse']:<10.4f} MAE: {result['mae']:<10.4f}")
                except NotImplementedError:
                    print(f"[!] {algo} not fully implemented yet, skipping...")
                except Exception as e:
                    print(f"[X] Error testing {algo}: {e}")
        
        # Test SimHash for Hamming distance
        if test_hamming and 'hamming' in test_metrics:
            print("\n" + "="*80)
            print("Testing Hamming Distance Estimation")
            print("="*80)
            for algo in test_hamming:
                print(f"\nTesting {algo}...")
                try:
                    result = self.test_hamming_distance(X, k, num_pairs, algo)
                    results.append(result)
                    print(f"[OK] MSE: {result['mse']:<10.4f} MAE: {result['mae']:<10.4f}")
                except NotImplementedError:
                    print(f"[!] {algo} not fully implemented yet, skipping...")
                except Exception as e:
                    print(f"[X] Error testing {algo}: {e}")
        
        # Test algorithms for Cosine Similarity
        if test_cosine and 'cosine' in test_metrics:
            print("\n" + "="*80)
            print("Testing Cosine Similarity Estimation")
            print("="*80)
            for algo in test_cosine:
                print(f"\nTesting {algo}...")
                try:
                    result = self.test_cosine_similarity(X, k, num_pairs, algo)
                    results.append(result)
                    print(f"[OK] MSE: {result['mse']:<10.4f} MAE: {result['mae']:<10.4f}")
                except NotImplementedError:
                    print(f"[!] {algo} not fully implemented yet, skipping...")
                except Exception as e:
                    print(f"[X] Error testing {algo}: {e}")
        
        return results


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Print header
    print(f"\n{'Algorithm':<12} {'Metric':<15} {'Sketch(s)':<10} {'Test(s)':<10} {'Total(s)':<10} {'MSE':<10} {'MAE':<10} {'RelErr':<20} {'Compress':<10}")
    print("-" * 115)
    
    # Sort results by Algorithm, then by Metric for cleaner grouping
    results.sort(key=lambda x: (x['algorithm'], x['metric']))
    
    # Print data
    for r in results:
        print(f"{r['algorithm']:<12} "
              f"{r['metric']:<15} "
              f"{r['sketch_time']:<10.4f} "
              f"{r['test_time']:<10.4f} "
              f"{r['total_time']:<10.4f} "
              f"{r['mse']:<10.4f} "
              f"{r['mae']:<10.4f} "
              f"{r['relative_error']:<20.4f} "
              f"{f'{r['compression_ratio']:.2f}x':<10}")


def compare_algorithms(results):
    """Compare algorithms and provide insights."""
    if not results:
        print("\n[!] No results to compare (no algorithms were successfully tested)")
        return
    
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON")
    print("="*80)
    
    # Find best algorithm for each metric
    best_mse = min(results, key=lambda x: x['mse'])
    best_mae = min(results, key=lambda x: x['mae'])
    fastest_sketch = min(results, key=lambda x: x['sketch_time'])
    fastest_test = min(results, key=lambda x: x['test_time'])
    
    print(f"\n[OK] Best MSE: {best_mse['algorithm']} ({best_mse['mse']:.4f})")
    print(f"[OK] Best MAE: {best_mae['algorithm']} ({best_mae['mae']:.4f})")
    print(f"[OK] Fastest Sketching: {fastest_sketch['algorithm']} ({fastest_sketch['sketch_time']:.4f}s)")
    print(f"[OK] Fastest Testing: {fastest_test['algorithm']} ({fastest_test['test_time']:.4f}s)")


def save_results_to_csv(results, filename=None):
    """Save results to file (JSON or CSV based on extension)."""
    if not results:
        print("\n[!] No results to save")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/test_results_{timestamp}.json'
    
    # Check extension and save accordingly
    if filename.endswith('.csv'):
        # Save as CSV
        try:
            # Get field names from the first result (assuming uniform keys)
            fieldnames = list(results[0].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[OK] Results saved to CSV: {filename}")
        except Exception as e:
            print(f"\n[X] Error saving to CSV: {e}")
            
    else:
        # Default to JSON
        if not filename.endswith('.json'):
            filename = filename.rsplit('.', 1)[0] + '.json'
            
        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(results, jsonfile, indent=2)
            print(f"\n[OK] Results saved to JSON: {filename}")
        except Exception as e:
            print(f"\n[X] Error saving to JSON: {e}")
            
    return filename


def main():
    """Run the complete testing pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test sketching algorithms on random sparse matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_pipeline.py                                  # Test all algorithms with default metrics
  python test_pipeline.py --algo BinSketch                 # Test only BinSketch
  python test_pipeline.py --algo BCS --metric inner_product # Test BCS with inner product
  python test_pipeline.py --metric jaccard                 # Test all algorithms with Jaccard
  python test_pipeline.py --algo BinSketch BCS             # Test BinSketch and BCS
  python test_pipeline.py --output results.json            # Save results to specific file
        """
    )
    parser.add_argument(
        '--algo', '--algorithm',
        nargs='+',
        choices=['BinSketch', 'BCS', 'MinHash', 'SimHash'],
        help='Specific algorithm(s) to test. If not specified, all algorithms will be tested.'
    )
    parser.add_argument(
        '--metric',
        nargs='+',
        choices=['inner_product', 'jaccard', 'hamming', 'cosine', 'all'],
        default=['all'],
        help='Similarity metric(s) to test. Options: inner_product, jaccard, hamming, cosine, all. Default: all'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON filename. If not specified, auto-generates filename with timestamp.'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-ALGORITHM TESTING PIPELINE")
    print("="*80)
    
    if args.algo:
        print(f"Testing algorithms: {', '.join(args.algo)}")
    else:
        print("Testing all available algorithms")
    
    if 'all' in args.metric:
        print(f"Testing all similarity metrics")
    else:
        print(f"Testing metrics: {', '.join(args.metric)}")
    
    # Configuration
    # Use random seed if not specified
    pipeline_seed = None 
    
    test_configs = [
        {
            'name': 'Tiny Sparse Matrix',
            'n_rows': 100,
            'n_cols': 100,
            'density': 0.1,
            'k': 50,
            'num_pairs': 50
        },
        {
            'name': 'Small Sparse Matrix',
            'n_rows': 1000,
            'n_cols': 1000,
            'density': 0.05,
            'k': 128,
            'num_pairs': 50
        },
        {
            'name': 'Large High-Dim Matrix',
            'n_rows': 2000,
            'n_cols': 100000, # 100k features
            'density': 0.001,
            'k': 256,
            'num_pairs': 20
        },
        {
            'name': 'Very Large High-Dim Matrix',
            'n_rows': 500, # Fewer rows to save memory
            'n_cols': 1000000, # 1 Million features
            'density': 0.0001,
            'k': 512,
            'num_pairs': 10
        }
    ]
    
    all_results = []  # Collect all results across configurations
    
    for idx, config in enumerate(test_configs, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST CONFIGURATION {idx}/{len(test_configs)}: {config['name']}")
        if config['n_cols'] >= 100000:
             print("(!) This test involves a very large matrix. It may take some time. Please wait...")
        print(f"{'#'*80}")
        
        # Generator seed (random per config if pipeline_seed is None)
        current_seed = np.random.randint(0, 100000) if pipeline_seed is None else pipeline_seed + idx
        
        # Generate random sparse matrix
        print(f"\nGenerating matrix: {config['n_rows']}x{config['n_cols']} "
              f"with {config['density']*100:.4f}% density...")
        X = generate_random_sparse_matrix(
            n_rows=config['n_rows'],
            n_cols=config['n_cols'],
            density=config['density'],
            seed=current_seed
        )
        print(f"[OK] Matrix generated: {X.nnz} non-zero elements")
        
        # Run pipeline with specified algorithms and metrics
        # Pass seed=None to AlgorithmTester to let it use internal randomness or controlled seed
        tester = AlgorithmTester(seed=current_seed)
        results = tester.run_pipeline(
            X, 
            k=config['k'], 
            num_pairs=config['num_pairs'],
            algorithms=args.algo,
            metrics=args.metric
        )
        
        # Add configuration name to results
        for result in results:
            result['config_name'] = config['name']
            result['density'] = config['density']
        
        all_results.extend(results)
        
        # Display results
        print_results_table(results)
        compare_algorithms(results)
    
    # Save all results to CSV
    save_results_to_csv(all_results, filename=args.output)
    
    print("\n\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
