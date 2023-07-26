import os
import json
from abc import abstractmethod
from configparser import ConfigParser


class Config:
    _instance = None
    _path = os.path.join(os.getcwd(), 'config.ini')
    _config = ConfigParser()
    _config.read(_path)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, section: str) -> None:
        self.section = section
        self.load_config()

    def get(self, arg: str) -> str:
        try:
            return self._config[self.section][arg]
        except:
            return None
    
    @staticmethod
    def get_config(section: str, arg: str):
        return json.loads(Config._config[section][arg])
    
    @abstractmethod
    def load_config():
        pass
    
class APIConfig(Config):
    def __init__(self) -> None:
        super().__init__('API')

    def load_config(self):
        self.host = self.get('host')
        self.port = int(self.get('port'))

class SYSConfig(Config):
    def __init__(self) -> None:
        super().__init__('SYS')

    def load_config(self):
        self.model_dir = os.getcwd() + self.get('model_dir')
        self.data_dir = os.getcwd() + self.get('data_dir')
        self.test_size_ratio = float(self.get('test_size_ratio'))
        self.random_state = int(self.get('random_state'))
        self.pool_size = int(self.get('pool_size'))
        self.drift_threshold = float(self.get('drift_threshold'))