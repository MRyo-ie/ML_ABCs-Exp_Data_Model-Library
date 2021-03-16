
import os.path as osp
from . import (  # .Experiment,  .
    # DataABC
        DataPPP,
        Raw_Data,
    # ModelABC
        ModelABC,
)


data_root_dir = osp.join('Experiment', 'data')


class Example_KaggleTemplate():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.dataPPP = DataPPP()
        self.dataPPP.load_data(self.train_path)

        self.clmns_conv_dict = {}
        self.clmns_XQA = {
            'X' : None,
            'Q' : None,
            'Y' : None,
        }

    def analyze_dataset(self):
        self.dataPPP.show_analyzed()
        self.dataPPP.show_info()
        # 確認して、XQA カラムリストを作る。→ initialize_dataPPP() で読み込む
    
    def initialize_dataPPP(self, 
            clmns_conv_dict={},# = {
                # 'date':'stream0', 
                # 'store':'class0', 'item':'class1',
                # 'sales':'num0',},
            clmns_XQA={}, #= {
                # 'X' : ['stream0', 'class0', 'class1',],
                # 'Q' : None,
                # 'Y' : ['num0'], }
            is_shuffle=True
        ):
        self.clmns_conv_dict = clmns_conv_dict
        self.clmns_XQA = clmns_XQA
        ###  train  ###
        self.dataPPP = self.build_dataPPP(self.train_path)
        self.dataPPP.split_train_valid_test(is_shuffle=is_shuffle)
        ###  test  ###
        dataPPP_test = self.build_dataPPP(self.test_path)
        self.dataPPP.X_test = dataPPP_test.X
        self.dataPPP.Q_test = dataPPP_test.Q
        self.dataPPP.Y_test = dataPPP_test.Y

    def build_dataPPP(self, path):
        dataPPP = DataPPP()
        dataPPP.load_data(path)
        dataPPP.df = dataPPP.df.rename(columns=self.clmns_conv_dict)

        # XQY に分解
        dataPPP.split_XQY(
            self.clmns_conv_dict,
            X_clmns=self.clmns_XQA['X'],
            Y_clmns=self.clmns_XQA['Y'],
            Q_clmns=self.clmns_XQA['Q'])
        return dataPPP

    def get_dataABC(self):
        return Raw_Data(self.dataPPP, exist_Q=False)




class Example_DemandForecasting(Example_KaggleTemplate):
    """https://qiita.com/simonritchie/items/8068a41765a9a43d0fea
    Prophet のやつ。
    """
    def __init__(self):
        super().__init__(
            osp.join(data_root_dir, 'demand-forecasting', 'train.csv'),
            osp.join(data_root_dir, 'demand-forecasting', 'test.csv')
        )
    
    def initialize_dataPPP(self, 
            store_item_id_tuple=(3,8),
            clmns_conv_dict = {
                'date':'stream0', 
                'store':'class0', 'item':'class1',
                'sales':'num0',},
            clmns_XQA = {
                'X' : ['stream0', 'class0', 'class1',],
                'Q' : None,
                'Y' : ['num0'], },
            is_shuffle=True,
        ):
        self.clmns_conv_dict = clmns_conv_dict
        self.clmns_XQA = clmns_XQA
        ###  train  ###
        self.dataPPP = self.build_dataPPP(self.train_path, store_item_id_tuple)
        self.dataPPP.split_train_valid_test(is_shuffle=is_shuffle)
        ###  test  ###
        dataPPP_test = self.build_dataPPP(self.test_path, store_item_id_tuple)
        self.dataPPP.X_test = dataPPP_test.X
        self.dataPPP.Y_test = dataPPP_test.Y

    def build_dataPPP(self, path, store_item_id_tuple):
        dataPPP = DataPPP()
        dataPPP.load_data(path)
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = dataPPP.df
        dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]

        dataPPP.df = dataPPP.df.rename(columns=self.clmns_conv_dict)
        # XQY に分解
        dataPPP.split_XQY(
            self.clmns_conv_dict,
            X_clmns=self.clmns_XQA['X'],
            Y_clmns=self.clmns_XQA['Y'],
            Q_clmns=self.clmns_XQA['Q'])
        return dataPPP



class Example_RestaurantRevenue(Example_KaggleTemplate):
    """https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
    LightGBMのやつ。
    I/O参考： https://www.kaggle.com/khanhdnguyen/restaurantrevenueprediction
    """
    def __init__(self):
        super().__init__(
            osp.join(data_root_dir, 'restaurant_revenue', 'train.csv.zip'),
            osp.join(data_root_dir, 'restaurant_revenue', 'test.csv.zip')
        )

    def initialize_dataPPP(self, 
            clmns_conv_dict = {
                'Open Date':'date0', 'City':'class0', 'City Group':'class1', 'Type':'class2',
                'P1':'num1', 'P2':'num2', 'P3':'num3', 'P4':'num4', 'P5':'num5', 'P6':'num6', 'P7':'num7', 'P8':'num8', 'P9':'num9', 'P10':'num10', 
                'P11':'num11', 'P12':'num12', 'P13':'num13', 'P14':'num14', 'P15':'num15', 'P16':'num16', 'P17':'num17', 'P18':'num18', 'P19':'num19', 'P20':'num20', 
                'P21':'num21', 'P22':'num22', 'P23':'num23', 'P24':'num24', 'P25':'num25', 'P26':'num26', 'P27':'num27', 'P28':'num28', 'P29':'num29', 'P30':'num30', 
                'P31':'num31', 'P32':'num32', 'P33':'num33', 'P34':'num34', 'P35':'num35', 'P36':'num36', 'P37':'num37',  'revenue':'num38'},
            clmns_XQA = {
                'X' : [
                    'date0', 'class0', 'class1', 'class2',
                    'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 
                    'num11', 'num12', 'num13', 'num14', 'num15', 'num16', 'num17', 'num18', 'num19', 'num20', 
                    'num21', 'num22', 'num23', 'num24', 'num25', 'num26', 'num27', 'num28', 'num29', 'num30', 
                    'num31', 'num32', 'num33', 'num34', 'num35', 'num36', 'num37', ],
                'Q' : None,
                'Y' : ['num38'], },
            is_shuffle=False,
        ):
        super().initialize_dataPPP(clmns_conv_dict, clmns_XQA, is_shuffle)




class Example_RecruitRestaurant():
    """https://www.codexa.net/kaggle-recruit-restaurant-visitor-forecasting-handson/#Kaggle-3
    平均値のやつ。
    """
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'recruit_restaurant', 'train.csv')
        self.test_path = osp.join(data_root_dir, 'recruit_restaurant', 'test.csv')

    def get_dataPPP(self):
        ###  train  ###
        self.dataPPP = self.build_dataPPP(self.train_path)
        self.dataPPP.X_train = self.dataPPP.X
        self.dataPPP.Y_train = self.dataPPP.Y
        # self.dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

        ###  test  ###
        dataPPP_test = self.build_dataPPP(self.test_path)
        self.dataPPP.X_test = dataPPP_test.X
        self.dataPPP.Y_test = dataPPP_test.Y

        return Raw_Data(self.dataPPP, exist_valid=False, exist_Q=False)

    def build_dataPPP(self, path, store_item_id_tuple=(3,2)):
        self.dataPPP = DataPPP()
        self.dataPPP.load_data(path)

        self.dataPPP.df = self.dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y0'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = self.dataPPP.df
        self.dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]
        # self.dataPPP.show_info()

        # XQY に分解
        self.dataPPP.split_XQY(
            self.clmns_conv_dict,
            X_clmns=['ds', 'store', 'item'],
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']

        return self.dataPPP







class Example_AshraeEnergy():
    def __init__(self):
        self.train_path = osp.join(data_root_dir, 'ashrae-energy', 'train.csv')
        self.test_path = osp.join(data_root_dir, 'ashrae-energy', 'test.csv')

    def get_dataPPP(self):
        ###  train  ###
        self.dataPPP = self.build_dataPPP(self.train_path)
        self.dataPPP.X_train = self.dataPPP.X
        self.dataPPP.Y_train = self.dataPPP.Y
        # self.dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

        ###  test  ###
        dataPPP_test = self.build_dataPPP(self.test_path)
        self.dataPPP.X_test = dataPPP_test.X
        self.dataPPP.Y_test = dataPPP_test.Y

        return Raw_Data(self.dataPPP, exist_valid=False, exist_Q=False)

    def build_dataPPP(self, path, store_item_id_tuple=(3,2)):
        self.dataPPP = DataPPP()
        self.dataPPP.load_data(path)

        self.dataPPP.df = self.dataPPP.df.rename(
                columns={'date': 'ds', 'sales': 'y0'})
        
        # ストアID = 1, 商品ID = 1  に限定して予測してみる。
        df = self.dataPPP.df
        self.dataPPP.df = df[(df.store == store_item_id_tuple[0]) & (df.item == store_item_id_tuple[1])]
        # self.dataPPP.show_info()

        # XQY に分解
        self.dataPPP.split_XQY(
            self.clmns_conv_dict,
            X_clmns=['ds', 'store', 'item'],
            Y_clmns=['y0'],
            Q_clmns=None)   # ['store', 'item']

        return self.dataPPP

