
import os.path as osp
from . import (  # .Experiment,  .
    # DataABC
        DataPPP,
        Raw_Data,
    # ModelABC
        ModelABC,
)



data_root_dir = osp.join('Experiment', 'data')

class Example_DemandForecasting():
    """https://qiita.com/simonritchie/items/8068a41765a9a43d0fea
    Prophet のやつ。
    """
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'demand-forecasting', 'train.csv')
        self.test_path = osp.join(data_root_dir, 'demand-forecasting', 'test.csv')

    def get_dataABC(self, store_item_id_tuple,
                    show_data_info=False):
        ###  train  ###
        dataPPP = self.get_dataPPP(self.train_path, store_item_id_tuple)
        dataPPP.X_train = dataPPP.X
        dataPPP.Y_train = dataPPP.Y
        # dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

        ###  test  ###
        dataPPP_test = self.get_dataPPP(self.test_path, store_item_id_tuple)
        dataPPP.X_test = dataPPP_test.X
        dataPPP.Y_test = dataPPP_test.Y

        if show_data_info:
            dataPPP.show_info()
            dataPPP.show_analyzed()
        dataPPP.show_analyzed()
        return Raw_Data(dataPPP, exist_valid=False, exist_Q=False)

    def get_dataPPP(self, path, store_item_id_tuple):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        dataPPP.df = dataPPP.df.rename(
                        columns = {
                            'date': 'date0', 
                            # 'store', 'item',
                            'sales': 'y0'
                        })
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]

        # XQY に分解
        dataPPP.split_XQY(
            X_clmns=['date0', 'store', 'item'],
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']

        return dataPPP



class Example_RestaurantRevenue():
    """https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
    LightGBMのやつ。
    I/O参考： https://www.kaggle.com/khanhdnguyen/restaurantrevenueprediction
    """
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'restaurant_revenue', 'train.csv.zip')
        self.test_path = osp.join(data_root_dir, 'restaurant_revenue', 'test.csv.zip')

        self.X_clmns_source = [
            'date0', 'City', 'City Group', 'Type',
            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 
            'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 
            'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 
            'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37']
        self.Y_clmns_source = ['revenue']
        self.Q_clmns_source = None

    def get_dataABC(self, 
                    X_clmns:list = None, 
                    Y_clmns:list = {'revenue'}, 
                    Q_clmns:list = None,
                    show_data_info=False):
        ###  train  ###
        dataPPP = self.get_dataPPP(self.train_path, X_clmns, Y_clmns, Q_clmns)
        # dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。
        dataPPP.X_train = dataPPP.X.copy()
        dataPPP.Y_train = dataPPP.Y.copy()

        ###  test  ###
        dataPPP_test = self.get_dataPPP(self.test_path, X_clmns, None, Q_clmns)
        dataPPP.X_test = dataPPP_test.X
        # dataPPP.Y_test = dataPPP_test.Y

        if show_data_info:
            dataPPP.show_info()
            dataPPP.show_analyzed()
        return Raw_Data(dataPPP, exist_valid=False, exist_Q=False)

    def get_dataPPP(self, path, 
                        X_clmns:list, Y_clmns:list=['revenue'],
                        Q_clmns:list=None):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        # カラム名を、XQY 形式に統一する。
        dataPPP.df = dataPPP.df.rename( 
                        columns={
                            'Open Date': 'date0', 
                            'revenue': 'y0',
                        })
        
        # XQY に分解
        dataPPP.split_XQY(
            X_clmns=X_clmns,
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']
        return dataPPP





class Example_RecruitRestaurant():
    """https://www.codexa.net/kaggle-recruit-restaurant-visitor-forecasting-handson/#Kaggle-3
    平均値のやつ。
    """
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'recruit_restaurant', 'train.csv')
        self.test_path = osp.join(data_root_dir, 'recruit_restaurant', 'test.csv')

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

    def get_dataPPP(self, path, store_item_id_tuple=(3,2)):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        dataPPP.df = dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y0'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]
        # dataPPP.show_info()

        # XQY に分解
        dataPPP.split_XQY(
            X_clmns=['ds', 'store', 'item'],
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']

        return dataPPP







class Example_AshraeEnergy():
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'ashrae-energy', 'train.csv')
        self.test_path = osp.join(data_root_dir, 'ashrae-energy', 'test.csv')

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

    def get_dataPPP(self, path, store_item_id_tuple=(3,2)):
        dataPPP = DataPPP()
        dataPPP.load_data(path)

        dataPPP.df = dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y0'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]
        # dataPPP.show_info()

        # XQY に分解
        dataPPP.split_XQY(
            X_clmns=['ds', 'store', 'item'],
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']

        return dataPPP

