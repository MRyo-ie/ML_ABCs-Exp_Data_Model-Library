import numpy as np
import pandas as pd 
import glob, re

from .abcs.abc_model import ModelABC


class DayOfTheWeek(ModelABC):
    """
    ・ 曜日の中央値を算出
        https://www.codexa.net/kaggle-recruit-restaurant-visitor-forecasting-handson/#Kaggle-3
    """
    def __init__(self):
        pass

    def fit(self, datadict):
        return super().fit(datadict)

    def predict(self, data):
        return super().predict(data)

    def predict_orgf(self, data):
        return super().predict_orgf(data)




