import os
import json
import pickle
import random
import logging
import pandas as pd
from core.model import Model
from lightgbm import LGBMClassifier

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)

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

    def init_config(self, phase: int, prob: int):
        self.name = f'phase{phase}_prob{prob}_model'
        self.logged_model = self.config.get(self.name)
        self.model_path = self.config.model_dir + '/' + self.name + '.pkl'
        self.label_encoder_path = self.config.model_dir + '/' + self.name + '_label_encoder.pkl'
        self.scaler_path = self.config.model_dir + '/' + self.name + '_scaler.pkl'
        self.train_data_path = self.config.data_dir + f'/phase-{phase}/prob-{prob}/raw_train.parquet'
        self.features_config : dict = json.loads(open(self.config.data_dir + f'/phase-{phase}/prob-{prob}/features_config.json', 'r').read())

    def load_encoder(self):
        try:
            self.label_encoder = pickle.load(open(self.label_encoder_path, 'rb'))
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))
        except:
            logger.info('Encoder not found, creating new one')
            self.label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.scaler = StandardScaler()
            
            data = pd.read_parquet(self.train_data_path, engine='pyarrow')
            data.drop("label", axis=1, inplace=True)
            data[self.features_config.get('category_columns', [])] = self.label_encoder.fit_transform(data[self.features_config.get('category_columns', [])])
            
            data = data[sorted(sorted(list(data.columns.values)))]
            print(data.columns)
            self.scaler.fit(data)

            os.makedirs(self.config.model_dir, exist_ok=True)
            pickle.dump(self.label_encoder, open(self.label_encoder_path, 'wb'))
            pickle.dump(self.scaler, open(self.scaler_path, 'wb'))

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.features_config.get('category_columns', [])] = self.label_encoder.transform(X[self.features_config.get('category_columns', [])])

        X = X[sorted(sorted(list(X.columns.values)))]
        print(X.columns)

        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        return X
    
    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)


def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_1.setup(2, 1)
    model_2.setup(2, 2)
    print('Phase 2\'s models loaded')