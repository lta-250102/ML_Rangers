import os
import json
from configparser import ConfigParser


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    # build path to config file
    path = os.path.join(os.getcwd(), 'config.ini')
    config = ConfigParser()
    config.read(path)

    def __init__(self, section: str) -> None:
        self.section = section

    def get(self, arg: str):
        return self.config[self.section][arg]
    
    @staticmethod
    def get_config(cls, section: str, arg: str):
        return json.loads(Config.config[section][arg])
    
class APIConfig(Config):
    def __init__(self) -> None:
        super().__init__('API')
        self.host = self.get('host')
        self.port = int(self.get('port'))

class SYSConfig(Config):
    def __init__(self) -> None:
        super().__init__('SYS')
        self.model_dir = os.getcwd() + self.get('model_dir')
        self.data_dir = os.getcwd() + self.get('data_dir')
        self.test_size_ratio = float(self.get('test_size_ratio'))
        self.random_state = int(self.get('random_state'))