import pickle
import pandas as pd
from core.model import Model
from core.config import SYSConfig
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


class Prob1Model(Model):
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        X = X.loc[:, ['feature3', 'feature11', 'feature15', 'feature16']]
        return X

class Prob2Model(Model):
    def train(self):
        '''train model'''
        self.load_encoder()
        super().train()

    def load_encoder(self):
        try:
            self.encoder = pickle.load(open(self.encoder_path, 'rb'))
        except:
            data = pd.read_parquet(self.train_data_path, engine='pyarrow')
            X = data.drop(columns=['label'])
            X_train, X_test = train_test_split(X, test_size=self.config.test_size_ratio, random_state=self.config.random_state)
            self.encoder = OrdinalEncoder()
            self.encoder.fit(X_train[self.category_columns])
            pickle.dump(self.encoder, open(self.encoder_path, 'wb'))

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        '''preprocess data'''
        tranformed = self.encoder.transform(X.loc[:, self.category_columns])
        X_cat = pd.DataFrame(tranformed, columns=self.category_columns)
        X_cleaned = X.drop(columns=self.category_columns, axis=1).reset_index(drop=True)
        X_cleaned = pd.concat([X_cleaned, X_cat], axis=1)
        return X_cleaned

    def load_model(self, phase: int, prob: int):
        '''load model'''
        self.encoder_name = f'/phase{phase}_prob{prob}_encoder.pkl'
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
        self.config = SYSConfig()
        self.encoder_path = self.config.model_dir + self.encoder_name
        self.load_encoder()
        super().load_model(phase, prob)


def load_model():
    '''load model'''
    model_1 = Prob1Model()
    model_2 = Prob2Model()
    model_1.load_model(1, 1)
    model_2.load_model(1, 2)
    print('Model loaded')