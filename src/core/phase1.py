import pickle
import pandas as pd
from core.config import SYSConfig
from api.request import Request
from api.response import Response
from sklearn.preprocessing import OrdinalEncoder
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


class Prob1Model():
    _instance = None
    name = '/phase1_prob1_model.pkl'
    train_data = '/phase-1/prob-1/raw_train.parquet'
    config = SYSConfig()
    model_path = config.model_dir + name
    train_data_path = config.data_dir + train_data
    excutor = ThreadPoolExecutor(max_workers=50)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Prob1Model, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    # def __init__(self) -> None:
    #     self.load_model()

    def train(self):
        '''train and save model'''
        # prepare data
        data = pd.read_parquet(self.train_data_path, engine='pyarrow')
        y = data['label']
        X = self.preprocess(data)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
        
        # train
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

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

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
        return X

    def load_model(self):
        try:
            # load model
            self.model = pickle.load(open(self.model_path, 'rb'))
        except:
            # init model
            print('Model not found, init model')
            self.model = RandomForestClassifier()
            self.train()


class Prob2Model():
    _instance = None
    name = '/phase1_prob2_model.pkl'
    encoder_name = '/phase1_prob2_encoder.pkl'
    train_data = '/phase-1/prob-2/raw_train.parquet'
    category_columns = [
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
    config = SYSConfig()
    encoder_path = config.model_dir + encoder_name
    model_path = config.model_dir + name
    train_data_path = config.data_dir + train_data
    excutor = ThreadPoolExecutor(max_workers=50)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Prob2Model, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    # def __init__(self) -> None:
    #     self.load_model()

    def train(self):
        '''train and save model'''
        # prepare data
        data = pd.read_parquet(self.train_data_path, engine='pyarrow')
        y = data['label']
        X = data.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
        self.encoder.fit(X_train[self.category_columns])

        X_train = self.preprocess(X_train)
        X_test = self.preprocess(X_test)
        
        # train
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

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

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        tranformed = self.encoder.transform(X.loc[:, self.category_columns])
        X_cat = pd.DataFrame(tranformed, columns=self.category_columns)
        X_cleaned = X.drop(columns=self.category_columns, axis=1).reset_index(drop=True)
        X_cleaned = pd.concat([X_cleaned, X_cat], axis=1)
        return X_cleaned

    def load_model(self):
        try:
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.encoder = pickle.load(open(self.encoder_path, 'rb'))
        except:
            # init model
            print('Model not found, init model')
            self.model = RandomForestClassifier()
            self.encoder = OrdinalEncoder()
            self.train()

