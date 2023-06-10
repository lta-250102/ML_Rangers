import mlflow
import pickle
import pandas as pd
from core.model import Model
from core.config import SYSConfig
from api.request import Request
from api.response import Response
from sklearn.preprocessing import OrdinalEncoder
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


class Prob1Model(Model):
    name = '/phase1_prob1_model.pkl'
    train_data = '/phase-1/prob-1/raw_train.parquet'
    config = SYSConfig()
    model_path = config.model_dir + name
    train_data_path = config.data_dir + train_data

    def train(self):
        '''train and save model'''
        # prepare data
        data = pd.read_parquet(self.train_data_path, engine='pyarrow')
        y = data['label']
        X = self.preprocess(data)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
        
        # train
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
        return X

    def load_model(self):
        try:
            self.model = pickle.load(open(self.model_path, 'rb'))
        except:
            self.train()

class Prob2Model(Model):
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

    def train(self):
        '''train and save model'''
        # prepare data
        data = pd.read_parquet(self.train_data_path, engine='pyarrow')
        y = data['label']
        X = data.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
        self.encoder = OrdinalEncoder()
        self.encoder.fit(X_train[self.category_columns])

        X_train = self.preprocess(X_train)
        X_test = self.preprocess(X_test)
        
        # train
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

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
            self.train()

def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    print('Model loaded')