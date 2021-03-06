import os
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
from fbprophet import Prophet

from .abcs.abc_model import ModelABC



class Prophet_Model(ModelABC):
    """
    ・ Prophet
        https://qiita.com/simonritchie/items/8068a41765a9a43d0fea
    ・ 因果推論
    """
    def __init__(self, model_instance=None):
        if model_instance is None:
            model_instance = Prophet()
        super().__init__(model_instance)
        self.name = 'Prophet'

    def fit(self, datadict):
        X = datadict['X']
        Y = datadict['Y']
        df = pd.concat([X, Y], axis=1)
        # print(df)
        self.model.fit(df=df)

    def predict(self, data):
        future_df = data['train']['X']
        forecast_df = self.model.predict(df=future_df)
        # print('\n\n\n----------------------------- predicted.')
        # print(forecast_df.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # 画像に保存する。
        os.makedirs('result', exist_ok=True)
        figure = self.model.plot(fcst=forecast_df)
        figure.savefig(fname='result/Prophet_result_1,1.png')
        return forecast_df['yhat']

    def predict_orgf(self, data):
        return super().predict_orgf(data)
