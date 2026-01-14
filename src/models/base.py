from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

class SketchModel(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def mapping(self, X: csr_matrix, k: int) -> csr_matrix:
        pass
    