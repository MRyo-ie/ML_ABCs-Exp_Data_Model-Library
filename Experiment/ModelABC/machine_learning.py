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
    def __init__(self, model_instance=None, 
                    params = {
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
        X = pd.concat([dataABC.X_train, dataABC.X_valid])   # .drop('stream0', axis=1)
        Y = pd.concat([dataABC.Y_train, dataABC.Y_valid])
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




# class RandomForest_Model(ModelABC):
#     """
#     ・ lightGBM（Gradient Boosting）
#         https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
#     ・ 因果推論
#     """
#     def __init__(self, model_instance=None, 
#                     params = {
#                         'learning_rate': [.01, .1, .5, .7, .9, .95, .99, 1],
#                         'boosting': ['gbdt'],
#                         'metric': ['l1'],
#                         'feature_fraction': [.3, .4, .5, 1],
#                         'num_leaves': [20],
#                         'min_data': [10],
#                         'max_depth': [10],
#                         'n_estimators': [10, 30, 50, 100]
#                     }):
#         lgb = lgbm.LGBMRegressor()
#         self.lgb_regressor = GridSearchCV(lgb, params, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
#         super().__init__(self.lgb_regressor)
#         self.name = 'LightGBM'
#         self.params = params

#     def fit(self, dataABC):
#         X = pd.concat([dataABC.X_train, dataABC.X_valid])   # .drop('stream0', axis=1)
#         Y = pd.concat([dataABC.Y_train, dataABC.Y_valid])
#         self.lgb_regressor.fit(X,Y)
#         self.model = lgbm.LGBMRegressor(
#                         learning_rate=self.lgb_regressor.best_params_["learning_rate"], 
#                         boosting='gbdt',  metric='l1', 
#                         feature_fraction=self.lgb_regressor.best_params_["feature_fraction"], 
#                         num_leaves=20, min_data=10, max_depth=10, 
#                         n_estimators=self.lgb_regressor.best_params_["n_estimators"], n_jobs=-1)
#         self.model.fit(X,Y)
#         # self.model = lgb.train(
#         #                 self.params,
#         #                 lgb_train,
#         #                 num_boost_round=100,
#         #                 valid_sets=lgb_valid,
#         #                 early_stopping_rounds=10)
#         print(f'Optimal lr: {self.lgb_regressor.best_params_["learning_rate"]}')
#         print(f'Optimal feature_fraction: {self.lgb_regressor.best_params_["feature_fraction"]}')
#         print(f'Optimal n_estimators: {self.lgb_regressor.best_params_["n_estimators"]}')
#         print(f'Best score: {self.lgb_regressor.best_score_}')

#     def predict(self, dataABC):
#         X_valid = dataABC.X_valid
#         result = self.model.predict(X_valid)
#         return pd.DataFrame(result, columns = ["pred"])

#     def predict_orgf(self, data):
#         return super().predict_orgf(data)




# ###    追加待ち    ###
# if __name__ == '__main__':
#     # https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
#     X_train, X_test, y_train, y_test = train_test_split(
#         x_, y_, random_state=0, train_size=0.7,shuffle=False)

#     # RandomForestRegressorによる予測
#     forest = RandomForestRegressor().fit(X_train, y_train)
#     prediction_rf = np.exp(forest.predict(test_feature))
#     acc_forest = forest.score(X_train, y_train)
#     acc_dic.update(model_forest = round(acc_forest,3))

#     # lasso回帰による予測
#     lasso = Lasso().fit(X_train, y_train)
#     prediction_lasso = np.exp(lasso.predict(test_feature))
#     acc_lasso = lasso.score(X_train, y_train)
#     acc_dic.update(model_lasso = round(acc_lasso,3))

#     # ElasticNetによる予測
#     En = ElasticNet().fit(X_train, y_train)
#     prediction_en = np.exp(En.predict(test_feature))
#     acc_ElasticNet = En.score(X_train, y_train)
#     acc_dic.update(model_ElasticNet = round(acc_ElasticNet,3))

#     # ElasticNetによるパラメータチューニング
#     parameters = {
#             'alpha'      : [0.001, 0.01, 0.1, 1, 10, 100],
#             'l1_ratio'   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     }

#     En2 = GridSearchCV(ElasticNet(), parameters)
#     En2.fit(X_train, y_train)
#     prediction_en2 = np.exp(En.predict(test_feature))
#     acc_ElasticNet_Gs = En2.score(X_train, y_train)
#     acc_dic.update(model_ElasticNet_Gs = round(acc_ElasticNet_Gs,3))

#     # 各モデルの訓練データに対する精度をDataFrame化
#     Acc = pd.DataFrame([], columns=acc_dic.keys())
#     dict_array = []
#     for i in acc_dic.items():
#             dict_array.append(acc_dic)
#     Acc = pd.concat([Acc, pd.DataFrame.from_dict(dict_array)]).T
#     Acc[0]



