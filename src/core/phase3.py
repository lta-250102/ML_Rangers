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

    def predict(self, columns: list[str], X: list[list]) -> list:
        try:
            X = pd.DataFrame(X, columns=columns)
            X = self.preprocess(X)
            y = self.model.predict(X)
            return y.tolist()
        except Exception as e:
            logger.exception(e)
            raise e
        
    def load_model(self):
        try:
            self.model = mlflow.pyfunc.load_model(self.logged_model)
        except:
            try:
                self.model = pickle.load(open(self.model_path, 'rb'))
            except:
                self.init_model()
                self.train()

    def init_config(self, phase: int, prob: int):
        self.name = f'phase{phase}_prob{prob}_model'
        self.logged_model = self.config.get(self.name)
        self.model_path = self.config.model_dir + '/' + self.name + '.pkl'
        self.cat_encoder_path = self.config.model_dir + '/' + self.name + '_cat_encoder.pkl'
        self.scaler_path = self.config.model_dir + '/' + self.name + '_scaler.pkl'
        self.train_data_path = self.config.data_dir + f'/phase-{phase}/prob-{prob}/raw_train.parquet'
        self.features_config : dict = json.loads(open(self.config.data_dir + f'/phase-{phase}/prob-{prob}/features_config.json', 'r').read())

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