import os
import json
import pickle
import random
import logging
import pandas as pd
from core.model import Model
from lightgbm import LGBMClassifier


logger = logging.getLogger("ml_ranger_logger")

class Prob1Model(Model):
    def init_model(self):
        self.model = LGBMClassifier()

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)

class Prob2Model(Model):
    def init_model(self):
        self.model = LGBMClassifier()

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)


def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_1.setup(3, 1)
    model_2.setup(3, 2)
    logger.info('Phase 3\'s models loaded')