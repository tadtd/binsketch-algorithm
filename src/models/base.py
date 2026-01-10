from abc import ABC, abstractmethod

class SketchModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def mapping(self, X):
        pass
    