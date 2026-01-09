from abc import ABC, abstractmethod

class SketchModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, X):
        pass
    