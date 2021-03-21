import os
import pandas as pd
import numpy as np

import lightgbm as lgbm
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import GridSearchCV
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
    
    def rename_clmns(self, dataABC):
        y_name = dataABC.clmns_XQA['Y'][0]
        return {'stream0': 'ds', y_name: 'y'}

    def fit(self, dataABC):
        X = pd.concat([dataABC.X_train, dataABC.X_valid])
        Y = pd.concat([dataABC.Y_train, dataABC.Y_valid])
        df = pd.concat([X, Y], axis=1)
        ## X.date0 -> ds,  Y.y0 -> y
        df = df.rename(columns=self.rename_clmns(dataABC))
        self.model.fit(df=df)

    def predict(self, dataABC):
        X = pd.concat([dataABC.X_train, dataABC.X_valid])
        future_df = X.rename(columns=self.rename_clmns(dataABC))
        # print('\n\nfuture_df : \n', future_df)
        forecast_df = self.model.predict(df=future_df)
        # print('\n\n\n----------------------------- predicted.')
        # print(forecast_df.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # 画像に保存する。
        os.makedirs('result', exist_ok=True)
        figure = self.model.plot(fcst=forecast_df)
        figure.savefig(fname='result/Prophet_result.png')
        return forecast_df['yhat']

    def predict_orgf(self, data):
        return super().predict_orgf(data)



class LightGBM_Model(ModelABC):
    """
    ・ lightGBM（Gradient Boosting）
        https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
    ・ 因果推論
    """
    def __init__(self, params = {
                        'learning_rate': [.01, .1, .5, .7, .9, .95, .99, 1],
                        'boosting': ['gbdt'],
                        'metric': ['l1'],
                        'feature_fraction': [.3, .4, .5, 1],
                        'num_leaves': [20],
                        'min_data': [10],
                        'max_depth': [10],
                        'n_estimators': [10, 30, 50, 100]
                    }):
        lgb = lgbm.LGBMRegressor()
        self.lgb_regressor = GridSearchCV(lgb, params, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
        super().__init__(self.lgb_regressor)
        self.name = 'LightGBM'
        self.params = params

    def fit(self, dataABC):
        X = pd.concat([dataABC.X_train, dataABC.X_valid]).to_numpy()
        Y = pd.concat([dataABC.Y_train, dataABC.Y_valid]).to_numpy().ravel()
        print(Y.shape)
        # import sys
        # sys.exit()
        self.lgb_regressor.fit(X,Y)

        self.model = lgbm.LGBMRegressor(
                        learning_rate=self.lgb_regressor.best_params_["learning_rate"], 
                        boosting='gbdt',  metric='l1', 
                        feature_fraction=self.lgb_regressor.best_params_["feature_fraction"], 
                        num_leaves=20, min_data=10, max_depth=10, 
                        n_estimators=self.lgb_regressor.best_params_["n_estimators"], n_jobs=-1)
        self.model.fit(X,Y)
        # self.model = lgb.train(
        #                 self.params,
        #                 lgb_train,
        #                 num_boost_round=100,
        #                 valid_sets=lgb_valid,
        #                 early_stopping_rounds=10)
        print(f'Optimal lr: {self.lgb_regressor.best_params_["learning_rate"]}')
        print(f'Optimal feature_fraction: {self.lgb_regressor.best_params_["feature_fraction"]}')
        print(f'Optimal n_estimators: {self.lgb_regressor.best_params_["n_estimators"]}')
        print(f'Best score: {self.lgb_regressor.best_score_}')

    def predict(self, dataABC):
        X_valid = dataABC.X_valid
        result = self.model.predict(X_valid)
        return pd.DataFrame(result, columns = ["pred"])

    def predict_orgf(self, data):
        return super().predict_orgf(data)



from sklearn.ensemble import RandomForestRegressor

class RandomForest_Model(ModelABC):
    """
    ・ lightGBM（Gradient Boosting）
        https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
    ・ 因果推論
    """
    def __init__(self, params = {
                        'max_depth': [10, 30, 35, 50, 65, 75, 100],
                        'max_features': [.3, .4, .5, .6],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [30, 50, 100, 200]
                    }):
        rf = RandomForestRegressor()
        self.rf_regressor = GridSearchCV(rf, params, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
        super().__init__(self.rf_regressor)
        self.name = 'RandomForest'
        self.params = params

    def fit(self, dataABC):
        X = pd.concat([dataABC.X_train, dataABC.X_valid]).to_numpy()
        Y = pd.concat([dataABC.Y_train, dataABC.Y_valid]).to_numpy().ravel()
        print(Y.shape)
        # import sys
        # sys.exit()
        self.rf_regressor.fit(X, Y)
    
        self.model = RandomForestRegressor(max_depth=self.rf_regressor.best_params_["max_depth"], 
                                        max_features=self.rf_regressor.best_params_["max_features"], 
                                        min_samples_leaf=self.rf_regressor.best_params_["min_samples_leaf"], 
                                        min_samples_split=self.rf_regressor.best_params_["min_samples_split"], 
                                        n_estimators=self.rf_regressor.best_params_["n_estimators"], 
                                        n_jobs=-1, oob_score=True)
        self.model.fit(X, Y)
    
    def predict(self, dataABC):
        X_valid = dataABC.X_valid
        result = self.model.predict(X_valid)
        return pd.DataFrame(result, columns = ["pred"])

    def predict_orgf(self, data):
        return super().predict_orgf(data)



from xgboost import XGBRegressor, plot_importance
class XGBoost_Model(ModelABC):
    """
    ・ XGBoost
    """
    def __init__(self, params = {
                        'learning_rate': [.1, .5, .7, .9, .95, .99, 1],
                        'colsample_bytree': [.3, .4, .5, .6],
                        'max_depth': [4],
                        'alpha': [3],
                        'subsample': [.5],
                        'n_estimators': [30, 70, 100, 200]
                    }):
        xgb_model = XGBRegressor()
        self.xgb_regressor = GridSearchCV(xgb_model, params, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
        super().__init__(self.xgb_regressor)
        self.name = 'XGBoost'
        self.params = params

    def fit(self, dataABC):
        # print(dataABC.X_valid.head())
        X = pd.concat([dataABC.X_train, dataABC.X_valid]).to_numpy()
        Y = pd.concat([dataABC.Y_train, dataABC.Y_valid]).to_numpy().ravel()
        print(Y.shape)
        # import sys
        # sys.exit()
        self.xgb_regressor.fit(X, Y)
    
        self.model = XGBRegressor(
                        learning_rate=self.xgb_regressor.best_params_["learning_rate"], 
                        colsample_bytree=self.xgb_regressor.best_params_["colsample_bytree"], 
                        max_depth=4, alpha=3, subsample=.5, 
                        n_estimators=self.xgb_regressor.best_params_["n_estimators"], n_jobs=-1)
        self.model.fit(X, Y)
    
    def predict(self, dataABC):
        X_valid = dataABC.X_valid
        result = self.model.predict(X_valid)
        return pd.DataFrame(result, columns = ["pred"])

    def predict_orgf(self, data):
        return super().predict_orgf(data)




