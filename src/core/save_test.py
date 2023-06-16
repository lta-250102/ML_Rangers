from api.response import Response
from api.request import Request
import pandas as pd
import os


def save_test(request: Request, response: Response, phase: str, prob: str):
    path = os.path.join(os.getcwd(), f'storage/request/phase{phase}_prob{prob}.csv')
    data = pd.DataFrame(request.rows, columns=request.columns)
    data['label'] = response.predictions
    if os.path.exists(path):
        old_data = pd.read_csv(path)
        data = pd.concat([old_data, data], axis=0)
    data.to_csv(path, index=False)
