import os
import pickle
import random
import logging
import pandas as pd
from core.model import Model
# from lightgbm import LGBMClassifier
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
from catboost import CatBoostClassifier

import json


logger = logging.getLogger("ml_ranger_logger")

class Prob1Model(Model):
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
        return X

    def init_model(self):
        self.model = CatBoostClassifier()

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)

class Prob2Model(Model):
    def load_encoder(self):
        try:
            self.encoder = pickle.load(open(self.encoder_path, 'rb'))

            with open('cache.json', 'r') as openfile:
                cache = json.load(openfile)
            self.object_features = cache.object_features
            self.onehot_features = cache.onehot_features
            self.scale_features = cache.scale_features
            
        except:
            data = pd.read_parquet(self.train_data_path, engine='pyarrow')
            X = data.drop(columns=['label'])

            self.object_features = list(X.select_dtypes(include=['object']).columns)

            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

            temp = pd.DataFrame(self.encoder.fit_transform(X[self.object_features]))
            temp.columns = self.encoder.get_feature_names_out(self.object_features)
            X.drop(columns=self.object_features ,axis=1, inplace=True)
            X = pd.concat([X, temp], axis=1)

            # X_train, _ = train_test_split(X, test_size=self.config.test_size_ratio, random_state=self.config.random_state)

            self.scaler = StandardScaler()

            self.onehot_features = [column for column in list(X.columns) if column.split('_')[0] in self.object_features]
            
            temp = X.drop(columns=self.onehot_features, axis=1, inplace=False)
            self.scale_features = list(temp.columns)

            self.scaler.fit(temp)

            cache = {
                'object_features': self.object_features,
                'onehot_features': self.onehot_features,
                'scale_features': self.scale_features
            }

            json_object = json.dumps(cache, indent=4)
            with open("cache.json", "w") as outfile:
                outfile.write(json_object)
            
            os.path.exists(self.config.model_dir) or os.makedirs(self.config.model_dir)
            pickle.dump(self.encoder, open(self.encoder_path, 'wb'))

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        try:
            # tranformed = self.encoder.transform(X.loc[:, self.category_columns])
            # X_cat = pd.DataFrame(tranformed, columns=self.category_columns)
            # # X_cleaned = X.drop(columns=self.category_columns.__add__(['batch_id', 'is_drift']), axis=1).reset_index(drop=True)
            # X_cleaned = X.loc[:, self.num_columns]
            # X_cleaned = pd.concat([X_cleaned, X_cat], axis=1)
            # return X_cleaned
            df = X.copy()

            redundant_features = ['batch_id', 'is_drift']
            for feature in redundant_features:
                if feature in list(df.columns):
                    df.drop(feature, inplace=True, axis=1)

            temp = pd.DataFrame(self.encoder.transform(df[self.object_features]))
            temp.columns = self.encoder.get_feature_names_out(self.object_features)
            df.drop(columns=self.object_features ,axis=1, inplace=True)
            df = pd.concat([df, temp], axis=1)

            temp = df.drop(columns=self.onehot_features, axis=1, inplace=False)
            temp = self.scaler.transform(temp)

            df.loc[:, self.scale_features] = temp
            return df

        except Exception as e:
            logger.exception(e)
            raise e
    
    def init_config(self, phase: int, prob: int):
        super().init_config(phase, prob)
        self.encoder_name = f'/phase{phase}_prob{prob}_encoder.pkl'
        self.num_columns = ["feature2", "feature5", "feature13", "feature18"]
        self.category_columns = [
            "feature1",
            "feature3",
            "feature4",
            "feature6",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
            "feature11",
            "feature12",
            "feature14",
            "feature15",
            "feature16",
            "feature17",
            "feature19",
            "feature20"
        ]
        self.encoder_path = self.config.model_dir + self.encoder_name

    def load_model(self):
        '''load model'''
        self.load_encoder()
        super().load_model()

    def init_model(self):
        self.model = CatBoostClassifier()

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)


def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_2.setup(1, 2)
    model_1.setup(1, 1)
    print('Model loaded')