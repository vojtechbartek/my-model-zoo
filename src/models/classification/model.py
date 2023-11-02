"""Abstract model"""

from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def fit():
        pass
    
    @abstractmethod
    def predict():
        pass
    
