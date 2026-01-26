from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np


class SketchModel(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def mapping(self, X: csr_matrix, k: int) -> np.ndarray:
        pass
    