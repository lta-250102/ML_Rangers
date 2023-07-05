import os
import json
import pickle
import mlflow
import logging
import pandas as pd
from abc import abstractmethod
from api.request import Request
from api.response import Response
from core.config import SYSConfig
from api.cache import CacheFinder
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ThreadPoolExecutor
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


logger = logging.getLogger("ml_ranger_logger")

class Model:
    _instance = None
    config = SYSConfig()
    excutor = ThreadPoolExecutor(max_workers=config.pool_size)
    cache_finder = CacheFinder()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance        

    def infer(self, request: Request) -> Response:
        cache_predictions, appeared_lst = self.cache_finder.find_appeared_lst(request)
        rows_to_predict = [row for row, appeared in zip(request.rows, appeared_lst) if not appeared]
        logger.info(f'Cache hit: {len(cache_predictions) - len(rows_to_predict)}')

        if len(rows_to_predict) > 0:
            prediction = self.excutor.submit(self.predict, request.columns, rows_to_predict)
            prediction_result = prediction.result()
            self.cache_finder.save_cache(request.rows, prediction_result)

            final_predictions = []
            for i in range(len(appeared_lst)):
                if appeared_lst[i]:
                    final_predictions.append(cache_predictions[i])
                else:
                    final_predictions.append(prediction_result.pop(0))
        else:
            final_predictions = cache_predictions

        return Response(
            id=request.id,
            predictions=final_predictions,
            drift=self.calculate_drift()
        )
    
    def predict(self, columns: list[str], X: list[list]) -> list:
        try:
            X = pd.DataFrame(X, columns=columns)
            X = self.preprocess(X)
            y = self.model.predict(X)
            return y.tolist()
        except Exception as e:
            logger.exception(e)
            raise e
    
    def setup(self, phase: int, prob: int):
        self.init_config(phase, prob)
        self.load_encoder()
        self.load_model()
    
    def load_encoder(self):
        try:
            self.cat_encoder = pickle.load(open(self.cat_encoder_path, 'rb'))
            self.scaler = pickle.load(open(self.scaler_path, 'rb'))
        except:
            logger.info('Encoder not found, creating new one')
            self.cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.scaler = StandardScaler()
            
            data = pd.read_parquet(self.train_data_path, engine='pyarrow')
            X_cat = data[self.features_config.get('category_columns', [])]
            X_num = data[self.features_config.get('numeric_columns', [])]
            self.cat_encoder.fit(X_cat)
            self.scaler.fit(X_num)

            os.makedirs(self.config.model_dir, exist_ok=True)
            pickle.dump(self.cat_encoder, open(self.cat_encoder_path, 'wb'))
            pickle.dump(self.scaler, open(self.scaler_path, 'wb'))

    def load_model(self):
        try:
            self.model = mlflow.pyfunc.load_model(self.logged_model)
        except:
            try:
                self.model = pickle.load(open(self.model_path, 'rb'))
            except:
                self.init_model()
                self.train()

    def train(self):
        try:
            data = pd.read_parquet(self.train_data_path, engine='pyarrow')
            y = data[self.features_config.get('target_column', 'label')]
            X = data.drop(columns=[self.features_config.get('target_column', 'label')])
            X_cleaned = self.preprocess(X)
            X_train, X_test, y_train, y_test = train_test_split(X_cleaned.values, y.values, test_size=self.config.test_size_ratio, random_state=self.config.random_state, stratify=y)
            
            with mlflow.start_run():
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
            
                # mlflow log
                mlflow.log_params(self.model.get_params())
                # mlflow.log_metrics(metrics={'accuracy': accuracy_score(y_test, y_pred),
                #                     "roc-auc": roc_auc_score(y_test, y_pred)})
                mlflow.log_metrics(metrics={'accuracy': accuracy_score(y_test, y_pred)})

                mlflow.sklearn.log_model(self.model, 'model', signature=infer_signature(X, y))
                mlflow.end_run()

                # save model
                pickle.dump(self.model, open(self.model_path, 'wb'))
        except Exception as e:
            logger.exception(e)
            raise e

    def init_config(self, phase: int, prob: int):
        self.name = f'phase{phase}_prob{prob}_model'
        self.logged_model = self.config.get(self.name)
        self.model_path = self.config.model_dir + '/' + self.name + '.pkl'
        self.cat_encoder_path = self.config.model_dir + '/' + self.name + '_cat_encoder.pkl'
        self.scaler_path = self.config.model_dir + '/' + self.name + '_scaler.pkl'
        self.train_data_path = self.config.data_dir + f'/phase-{phase}/prob-{prob}/raw_train.parquet'
        self.features_config : dict = json.loads(open(self.config.data_dir + f'/phase-{phase}/prob-{prob}/features_config.json', 'r').read())

    @abstractmethod
    def init_model(self):
        pass

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_cat = X[self.features_config.get('category_columns', [])]
        X_num = X[self.features_config.get('numeric_columns', [])]

        X_cat = pd.DataFrame(self.cat_encoder.transform(X_cat), columns=self.cat_encoder.get_feature_names_out(self.features_config.get('category_columns', [])))
        X_num = pd.DataFrame(self.scaler.transform(X_num), columns=self.features_config.get('numeric_columns', []))

        X = pd.concat([X_cat, X_num], axis=1)
        return X

    @abstractmethod
    def calculate_drift(self) -> int:
        pass