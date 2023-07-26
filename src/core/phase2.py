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
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 0,
        }
        self.model = LGBMClassifier()


class Prob2Model(Model):
    def init_model(self):
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 0,
        }
        self.model = LGBMClassifier()



def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_1.setup(2, 1)
    model_2.setup(2, 2)
    print('Phase 2\'s models loaded')