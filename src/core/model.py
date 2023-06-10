import pandas as pd
from abc import abstractmethod
from api.request import Request
from api.response import Response
from concurrent.futures import ThreadPoolExecutor


class Model:
    _instance = None
    excutor = ThreadPoolExecutor(max_workers=50)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance.load_model()
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
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def train(self):
        pass