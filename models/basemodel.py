from abc import ABC, abstractmethod
from utils.config import Config
class BaseModel(ABC):
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

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