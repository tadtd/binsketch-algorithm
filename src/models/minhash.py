from .base import SketchModel
import numpy as np
from scipy.sparse import csr_matrix

class MinHash(SketchModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def mapping(self, X: csr_matrix, k: int) -> csr_matrix:
        """
        Projects high-dimensional binary data into a lower-dimensional MinHash sketch.
        Args:
            X: Input sparse matrix (n_samples, n_features).
            k: Target dimension for the sketch.
        Returns:
            Sketch matrix where each row contains k MinHash signatures.
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(seed=self.seed)
        
        # Generate k different hash permutations
        sketches = []
        for i in range(n_samples):
            row = X.getrow(i)
            nonzero_indices = row.nonzero()[1]
            
            if len(nonzero_indices) == 0:
                # If the row is empty, sketch is all zeros
                sketch_row = np.zeros(k, dtype=X.dtype)
            else:
                sketch_row = np.zeros(k, dtype=np.int32)
                # For each hash function
                for j in range(k):
                    # Create a permutation based on seed + j
                    perm_rng = np.random.RandomState(seed=self.seed + j)
                    permutation = perm_rng.permutation(n_features)
                    
                    # Find the minimum hash value among nonzero positions
                    min_hash = n_features
                    for idx in nonzero_indices:
                        hash_val = np.where(permutation == idx)[0][0]
                        if hash_val < min_hash:
                            min_hash = hash_val
                    
                    sketch_row[j] = min_hash
            
            sketches.append(sketch_row)
        
        # Convert to sparse matrix
        sketch_matrix = csr_matrix(np.array(sketches), dtype=X.dtype)
        return sketch_matrix
    
    def estimate_jaccard_similarity(self, sketch1: csr_matrix, sketch2: csr_matrix) -> float:
        """
        Estimates Jaccard similarity between two MinHash sketches.
        Args:
            sketch1: First MinHash sketch.
            sketch2: Second MinHash sketch.
        Returns:
            Estimated Jaccard similarity (between 0 and 1).
        """
        if sketch1.shape != sketch2.shape:
            raise ValueError("Sketches must have the same shape for Jaccard similarity estimation.")
        # Convert to dense arrays for comparison
        sig1 = sketch1.toarray().ravel()
        sig2 = sketch2.toarray().ravel()
        
        # Count how many hash values match
        matches = np.sum(sig1 == sig2)
        
        # Jaccard similarity estimate is the fraction of matching signatures
        jaccard_sim = matches / len(sig1)
        
        return jaccard_sim