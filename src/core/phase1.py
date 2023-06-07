import pickle
import pandas as pd
from core.config import SYSConfig
from api.request import Phase1Prob1Request
from api.response import Phase1Prob1Response
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


class Prob1Model():
    _instance = None
    name = '/phase1_prob1.pkl'
    train_data = '/phase-1/prob-1/raw_train.parquet'
    config = SYSConfig()
    model_path = config.model_dir + name
    train_data_path = config.data_dir + train_data

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Prob1Model, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        self.excutor = ThreadPoolExecutor(max_workers=15)
        self.load_model()

    def train(self):
        '''train and save model'''
        # prepare data
        data = pd.read_parquet(Prob1Model.train_data_path, engine='pyarrow')
        y = data['label']
        X = self.preprocess(data)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Prob1Model.config.test_size_ratio, random_state=Prob1Model.config.random_state)
        
        # train
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # save model
        pickle.dump(self.model, open(Prob1Model.model_path, 'wb'))

    def infer(self, request: Phase1Prob1Request) -> Phase1Prob1Response:
        prediction = self.excutor.submit(self.predict, request.columns, request.rows)
        response = Phase1Prob1Response(
            id=request.id,
            prediction=prediction.result(),
            drift=0
        )
        return response

    def predict(self, columns: list[str], X: list[list]) -> list:
        X = pd.DataFrame(X, columns=columns)
        X = self.preprocess(X)
        y = self.model.predict(X)
        return y.tolist()

    def preprocess(self, X: pd.DataFrame) -> list:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
        return X

    def load_model(self):
        try:
            # load model
            self.model = pickle.load(open(Prob1Model.model_path, 'rb'))
        except:
            # init model
            print('Model not found, init model')
            self.model = RandomForestClassifier()
            self.train()
