import pickle
import mlflow
import pandas as pd
from abc import abstractmethod
from api.request import Request
from api.response import Response
from core.config import SYSConfig
from concurrent.futures import ThreadPoolExecutor
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split


class Model:
    _instance = None
    config = SYSConfig()
    excutor = ThreadPoolExecutor(max_workers=config.pool_size)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance
    
    def infer(self, request: Request) -> Response:
        prediction = self.excutor.submit(self.predict, request.columns, request.rows)
        response = Response(
            id=request.id,
            # prediction=self.predict(request.columns, request.rows),
            prediction=prediction.result(),
            drift=0
        )
        return response
    
    def predict(self, columns: list[str], X: list[list]) -> list:
        X = pd.DataFrame(X, columns=columns)
        X = self.preprocess(X)
        y = self.model.predict(X)
        return y.tolist()
    
    def load_model(self, phase: int, prob: int):
        self.name = f'/phase{phase}_prob{prob}_model.pkl'
        self.train_data = f'/phase-{phase}/prob-{prob}/raw_train.parquet'
        self.model_path = self.config.model_dir + self.name
        self.train_data_path = self.config.data_dir + self.train_data
        try:
            self.model = pickle.load(open(self.model_path, 'rb'))
        except:
            self.train()
        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def train(self):
        data = pd.read_parquet(self.train_data_path, engine='pyarrow')
        y = data['label']
        X = data.drop(columns=['label'])
        X = self.preprocess(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
        
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            
            # mlflow log
            mlflow.log_params(self.model.get_params())
            mlflow.log_metrics({'accuracy': self.model.score(X_test, y_test)})
            mlflow.sklearn.log_model(self.model, 'model', signature=infer_signature(X_train, y_train))
            mlflow.end_run()

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
