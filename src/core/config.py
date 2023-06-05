import os
import json
from configparser import ConfigParser


class Config:
    # build path to config file
    path = os.path.join(os.getcwd(), 'config.ini')
    config = ConfigParser()
    config.read(path)

    def __init__(self, section: str) -> None:
        self.section = section

    def get(self, arg: str):
        return json.loads(self.config[self.section][arg])
    
    @staticmethod
    def get_config(cls, section: str, arg: str):
        return json.loads(Config.config[section][arg])