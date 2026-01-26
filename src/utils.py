import numpy as np

def save_compression_matrix(matrix: np.ndarray, output_path: str):
    np.save(output_path, matrix)