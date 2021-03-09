import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod



class DataPPP():
    """
    (Abstruct) ML Data PreProcess Pattern.

    実験（学習 → score, qini曲線）に使用する train/test データの生成をパターン化する。
    バラバラだったので、統一しやすいように規格化した。
    
    【目標】
        datasetの前処理 を抽象化する。
            ・ X_train, X_val, Q_train, Q_val, A_train, At_val,  に統合・分割する。
            ・ データ構造ごとに、一般的な形でデータを整形する。
                ・ .csv → pd.DataFrame
                ・ json, xml → dict, list
                ・ 画像 → 
                ・ 文章 → 
    """
    def __init__(self):
        # all
        self.dir_path = None
        self.df = None
        self.X = None
        self.Q = None
        self.Y = None
        # train
        self.X_train = None
        self.Q_train = None
        self.Y_train = None
        # valid
        self.X_valid = None
        self.Q_valid = None
        self.Y_valid = None
        # test
        self.X_test = None
        self.Q_test = None
        self.Y_test = None
    
    def load_data(self, path: str, X_clms: list = None, Y_clms: list = None, Q_clms: list = None, n=None):
        f_name, ext = osp.splitext(path)
        self.dir_path = Path(f_name).resolve().parents[0]
        if ext == '.csv':
            self.df = pd.read_csv(path)
        else:
            raise Exception(f"[Error] 対応していないデータ形式です。: {ext}")

        if X_clms is not None or Y_clms is not None or Q_clms is not None:
            self.split_XQY(X_clms, Y_clms, Q_clms)

    def show_info(self, df=None):
        """
        この辺の情報をもとに、データの前処理などを行う。
        """
        if df is None:
            df = self.df
        print("\n[Info] ============= head, tail")
        print(df.head())
        print(df.tail())
        print("\n[Info] ============= info")
        df.info()
        print("\n[Info] ============= describe")
        print(df.describe(), '\n')
    
    def split_XQY(self, X_clms: list = None, Y_clms: list = None, Q_clms: list = None):
        self.X = pd.DataFrame(self.df, columns=X_clms)
        self.Q = pd.DataFrame(self.df, columns=Q_clms)
        self.Y = pd.DataFrame(self.df, columns=Y_clms)

    def split_train_valid_test(self, do_valid=True, do_test=True, valid_size=0.2, test_size=0.2, random_state=30, is_shuffle=True):
        # 時系列データの場合は、分割、シャッフルはダメ。
        # 考える必要あり。
        if do_valid:
            self.X_train, self.X_valid, self.Q_train, self.Q_valid, self.Y_train, self.Y_valid = \
                train_test_split(
                    self.X_train.values, self.Q_train.values, self.Y_train.values, 
                    test_size=valid_size, random_state=random_state,
                    shuffle=is_shuffle, stratify=self.Y_train.values)
        if do_test:
            self.X_train, self.X_test, self.Q_train, self.Q_test, self.Y_train, self.Y_test = \
                train_test_split(
                    self.X.values, self.Q.values, self.Y.values, 
                    test_size=test_size, random_state=random_state, 
                    shuffle=is_shuffle, stratify=self.Y.values)
        # self.X_train, self.X_test, self.Q_train, self.Q_test, self.Y_train, self.Y_test = \
        #     train_test_split(
        #         self.X.values, self.Q.values, self.Y.values, 
        #         test_size=1 / 3, random_state=30, stratify=self.Y_train
        #     )

    def IPW(self):
        pass




class DataABC(metaclass=ABCMeta):
    """
    (Abstruct) ML Dataset Frame.

    機械学習 Experiment（train, test）に使用する train, test データの共通規格。
    ExperimentABC, ModelABC の Input の規格(config に相当)。
    
    【目標】
        datasetの型 を抽象化する。
        用途：
            ・ train/valid/test および X/Q/Y の整理
                ・ X : features。問題文。
                ・ Q : 状態、状況、条件。質問文。（タスクごとに変化。ドメインによる差の吸収に使う）
                    ・ P(X|Q) = Y
                    例） 
                    ・ VQAのQ（質問文）
                    ・ 強化学習の状態s、Uplift modelingのtreat、
                    ・ Modelが統計手法の場合は、基本的にQは使わない。（大抵の場合、Qはドメイン特有になるため。そこも予測するのが機械学習）
                ・ Y : 
            ・ Model や 前処理プログラムの Input 規格の統一。
                ・ ModelABC を継承する、機械学習モデル のInput。
                ・ バイアス除去（IPTW, DR, SDR） のInput。
                ・ ノイズ除去 のInput。
                ・ 異常検知（？） のInput。

    【実装例】
        class Data_UpliftBasic(UpliftModelTmpl):
            def get_train(self):
                ・・・
            def get_eval(self):
                ・・・
    """
    def __init__(self, dataPPP: DataPPP, exist_valid=True, exist_Q=True):
        # all
        self.dataPPP = dataPPP
        self.X = dataPPP.X.copy()
        self.Q = dataPPP.Q.copy()
        self.Y = dataPPP.Y.copy()

        ##  minimum  ##
        # train
        self.X_train = dataPPP.X_train.copy()
        self.Y_train = dataPPP.Y_train.copy()
        # test
        self.X_test = dataPPP.X_test.copy()
        self.Y_test = dataPPP.Y_test.copy()

        ##  Maximum  ##
        self.X_valid = None
        self.Y_valid = None
        self.Q_train = None
        self.Q_test = None
        self.Q_valid = None
        if exist_valid:
            # valid
            self.X_valid = dataPPP.X_valid.copy()
            self.Y_valid = dataPPP.Y_valid.copy()
        if exist_Q:
            # Q
            self.Q_train = dataPPP.Q_train.copy()
            self.Q_test = dataPPP.Q_test.copy()
        if exist_valid and exist_Q:
            self.Q_valid = dataPPP.Q_valid.copy()

        # data の型など、問題がないかチェックする。（未）
        # self._check()

    @abstractmethod
    def get_train(self):  #=> dect or stream
        """
        例）
            X = self.X_train
            y = self.Q_train * self.Y_train + (1 - self.Q_train) * (1 - self.Y_train)
            return {
                'X' : X,
                'y' : y,
            }
        """
        raise NotImplementedError()

    @abstractmethod
    def get_eval(self):  #=> dect or stream
        """
        例）
            return {
                'train' : {
                    'X': self.X_train, 
                    'Q': self.Q_train, 
                    'Y': self.Y_train,
                },
                'test' : { 
                    'X': self.X_test, 
                    'Q': self.Q_test, 
                    'Y': self.Y_test,
                }
            }
        """
        
    
    def _check(self, data):
        raise NotImplementedError()




