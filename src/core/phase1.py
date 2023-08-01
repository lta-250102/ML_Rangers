import os
import json
import pickle
import random
import logging
import pandas as pd
from core.model import Model
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


logger = logging.getLogger("ml_ranger_logger")

class Prob1Model(Model):
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
    #     # Preprocessing pipeline for numeric features
    #     X_train = X.copy()
    #     num_features = ['feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
    #             'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
    #             'feature15', 'feature16']
    #     num_transformer = StandardScaler()

    # # ColumnTransformer to apply different preprocessing steps to different feature types
    #     preprocessor = ColumnTransformer(
    #     transformers=[
    #     ('num', num_transformer, num_features)
    # ])

    # # Preprocess the training data
    #     X_train_preprocessed = preprocessor.fit_transform(X_train)
    #     X_train_preprocessed = pd.DataFrame(X_train_preprocessed)
    #     return X_train_preprocessed
        return X

    def init_model(self):
        params = {
        'learning_rate': 0.1,
        'n_estimators': 300,
        'max_depth': 5,
        'min_child_samples': 30,
        'reg_alpha': 0.5,
        'reg_lambda': 0,
}
        self.model = LGBMClassifier(params)

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)

class Prob2Model(Model):
    def load_encoder(self):
        try:
            self.encoder = pickle.load(open(self.encoder_path, 'rb'))
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))

            with open(os.path.join(self.config.data_dir, 'phase-1/prob-2/cache.json'), 'r') as openfile:
                cache = json.load(openfile)
            self.object_features = cache['object_features']
            self.onehot_features = cache['onehot_features']
            self.scale_features = cache['scale_features']

            with open(os.path.join(self.config.data_dir, 'phase-1/prob-2/features-prob2.json'), 'r') as openfile:
                rfe = json.load(openfile)
            self.rfe_features = rfe['selected_features']
            
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
            with open(os.path.join(self.config.data_dir, 'phase-1/prob-2/cache.json'), "w") as outfile:
                outfile.write(json_object)

            with open(os.path.join(self.config.data_dir, 'phase-1/prob-2/features-prob2.json'), 'r') as openfile:
                rfe = json.load(openfile)
            self.rfe_features = rfe['selected_features']
            
            os.path.exists(self.config.model_dir) or os.makedirs(self.config.model_dir)
            pickle.dump(self.encoder, open(self.encoder_path, 'wb'))
            pickle.dump(self.scaler, open(self.encoder_path, 'wb'))

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        try:
            # remove redundant features
            X.drop(columns=['batch_id', 'is_drift'], axis=1, errors='ignore', inplace=True)

            # one-hot encode object dtype features
            temp = pd.DataFrame(self.encoder.transform(X[self.object_features]))
            temp.columns = self.encoder.get_feature_names_out(self.object_features)
            X.drop(columns=self.object_features, axis=1, inplace=True)
            X = pd.concat([X, temp], axis=1)

            # scale down other features
            temp = X.drop(columns=self.onehot_features, axis=1, inplace=False)
            temp = self.scaler.transform(temp)

            X.loc[:, self.scale_features] = temp

            # get features that RFE outputs
            X = X.loc[:, self.rfe_features]
            return X

        except Exception as e:
            logger.exception(e)
            raise e
    
    def init_config(self, phase: int, prob: int):
        super().init_config(phase, prob)
        self.encoder_name = f'/phase{phase}_prob{prob}_encoder.pkl'
        self.scaler_name = f'/phase{phase}_prob{prob}_scaler.pkl'
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
        self.scaler_path = self.config.model_dir + self.scaler_name

    def load_model(self):
        '''load model'''
        self.load_encoder()
        super().load_model()

    def init_model(self):
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 0,
        }
        self.model = LGBMClassifier(params)

    def calculate_drift(self) -> int:
        '''calculate drift'''
        return random.randint(0, 1)


def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_1.setup(1, 1)
    model_2.setup(1, 2)
    logger.info('Phase 1\'s model loaded')