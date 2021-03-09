
import os.path as osp
from . import (  # .Experiment,  .
    # DataABC
        DataPPP,
        Raw_Data,
    # ModelABC
        ModelABC,
)


class Example_DemandForecasting():
    def __init__(self):
        self.train_path = osp.join('Experiment', 'data', 'demand-forecasting', 'train.csv')
        self.test_path = osp.join('Experiment', 'data', 'demand-forecasting', 'test.csv')

    def get_dataABC(self, limit_d=(1,1)):
        ###  train  ###
        dataPPP = self.get_dataPPP(self.train_path, limit_d)
        dataPPP.X_train = dataPPP.X
        dataPPP.Y_train = dataPPP.Y
        # dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

        ###  test  ###
        dataPPP_test = self.get_dataPPP(self.test_path, limit_d)
        dataPPP.X_test = dataPPP_test.X
        dataPPP.Y_test = dataPPP_test.Y

        return Raw_Data(dataPPP, exist_valid=False, exist_Q=False)

    def get_dataPPP(self, path, limit_d):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        dataPPP.df = dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == limit_d[0]) & (df.item == limit_d[1])]
        # dataPPP.show_info()

        # XQY に分解
        dataPPP.split_XQY(
            X_clms=['ds', 'store', 'item'],
            Y_clms=['y'],
            Q_clms=None)   # ['store', 'item']

        return dataPPP




class Example_AshraeEnergy():
    def __init__(self):
        self.train_path = osp.join('Experiment', 'data', 'ashrae-energy', 'train.csv')
        self.test_path = osp.join('Experiment', 'data', 'ashrae-energy', 'test.csv')

    def get_dataABC(self):
        ###  train  ###
        dataPPP = self.get_dataPPP(self.train_path)
        dataPPP.X_train = dataPPP.X
        dataPPP.Y_train = dataPPP.Y
        # dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

        ###  test  ###
        dataPPP_test = self.get_dataPPP(self.test_path)
        dataPPP.X_test = dataPPP_test.X
        dataPPP.Y_test = dataPPP_test.Y

        return Raw_Data(dataPPP, exist_valid=False, exist_Q=False)

    def get_dataPPP(self, path, limit_d=(3,2)):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        dataPPP.df = dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == limit_d[0]) & (df.item == limit_d[1])]
        # dataPPP.show_info()

        # XQY に分解
        dataPPP.split_XQY(
            X_clms=['ds', 'store', 'item'],
            Y_clms=['y'],
            Q_clms=None)   # ['store', 'item']

        return dataPPP

