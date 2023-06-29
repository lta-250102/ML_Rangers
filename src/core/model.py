import pickle
import mlflow
import logging
import pandas as pd
from abc import abstractmethod
from api.request import Request
from api.response import Response
from core.config import SYSConfig
from api.cache import CacheFinder
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split


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
            drift=0
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
        self.load_model()
    
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
            y = data['label']
            X = data.drop(columns=['label'])
            X = self.preprocess(X)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
            
            with mlflow.start_run():
                self.model.fit(X, y, verbose=False)
                
                # mlflow log
                mlflow.log_params(self.model.get_params())
                mlflow.log_metrics({'accuracy': self.model.score(X, y), 
                                    "roc-auc": roc_auc_score(y, self.model.predict(X))})

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
        self.train_data = f'/phase-{phase}/prob-{prob}/raw_train.parquet'
        self.model_path = self.config.model_dir + '/' + self.name + '.pkl'
        self.train_data_path = self.config.data_dir + self.train_data

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def calculate_drift(self) -> int:
        pass