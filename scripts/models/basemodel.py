from abc import ABC, abstractmethod
from scripts.utils.config import Config
class BaseModel(ABC):
    def __init__(self, cfg, base_model):
        self.config = Config.from_json(cfg)
        self.base_model = base_model

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def compile(self):
        pass


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass