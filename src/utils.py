"""Utility functions for data processing."""
import numpy as np


def save_compression_matrix(matrix: np.ndarray, output_path: str):
    """Save compression matrix to file.
    
    Args:
        matrix: Numpy array to save
        output_path: Path where the matrix will be saved
    """
    np.save(output_path, matrix)
