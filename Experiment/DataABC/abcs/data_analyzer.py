import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .abc_data import DataPPP
import warnings
warnings.filterwarnings("ignore")


class DataAnalyzer():
    def __init__(self, dataPPP:DataPPP):
        self.dataPPP = dataPPP

    ###   show   ###
    def show_info(self, df=None):
        if df is None:
            df = self.dataPPP.df
        print("\n[Info] info =============================================")
        df.info()
        print("\n[Info] describe =========================================")
        print(df.describe(), '\n')
    
    def show_analyzed(self, df=None):
        if df is None:
            df = self.dataPPP.df
        print("\n[Info] feature analyses (numerical, categorical) ========")
        print(df.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist())
        print(df.select_dtypes([np.number]).columns.tolist())
        print("\n[Info] head, tail =======================================")
        print(df.head())
        print(df.tail())

    def show_counts(self):
        print(f"""Shapes: 
            X_train  {self.dataPPP.X_train.shape} 
            X_valid  {self.dataPPP.X_valid.shape} 
            X_test   {self.dataPPP.X_test.shape} 
            Y_train  {self.dataPPP.Y_train.shape} 
            Y_valid  {self.dataPPP.Y_valid.shape}
            Y_test   {self.dataPPP.Y_test.shape} """)

    def plot_numerical_dist(self, data_name:str=''):
        X_train = self.dataPPP.X_train
        Y_train = self.dataPPP.Y_train
        # numerical（int, floatなど）の可視化
        numerical_features = X_train.select_dtypes([np.number]).columns.tolist()

        n = len(X_train[numerical_features].columns)
        w = 3
        h = (n - 1) // w + 1
        fig, axes = plt.subplots(h, w, figsize=(w * 6, h * 3))
        for i, (name, col) in enumerate(X_train[numerical_features].items()):
            r, c = i // w, i % w
            ax = axes[r, c]
            col.hist(ax=ax)
            ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)
            ax2.set_ylim(0)

        fig.tight_layout()
        plt.savefig(f'result/{data_name}_numerical_dist_X.png')

        plt.clf()
        print(Y_train.describe())
        sns.distplot(a=Y_train, kde=True).set(xlabel='revenue', ylabel='P(revenue)')
        plt.savefig(f'result/{data_name}_numerical_dist_Y.png')

    def plot_categorical_dist(self, data_name:str=''):
        df = self.dataPPP.df
        Y_clmn_name = self.dataPPP.clmns_XQA['Y'][0]
        print(f'\n\n[Info] Y_clmn_name : {Y_clmn_name}')
        categorical_features = df.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()
        fig, ax = plt.subplots(3, 1, figsize=(40, 30))
        for variable, subplot in zip(categorical_features, ax.flatten()):
            df_2 = df[[variable,Y_clmn_name]].groupby(variable).revenue.sum().reset_index()
            df_2.columns = [variable,'total_revenue']
            sns.barplot(x=variable, y='total_revenue', data=df_2 , ax=subplot)
            subplot.set_xlabel(variable,fontsize=20)
            subplot.set_ylabel('Total Revenue',fontsize=20)
            for label in subplot.get_xticklabels():
                label.set_rotation(45)
                label.set_size(20)
            for label in subplot.get_yticklabels():
                label.set_size(20)
        fig.tight_layout()
        plt.savefig(f'result/{data_name}_categorical_dist_X.png')


    # サンプルから欠損値と割合、データ型を調べる関数
    def missing_table(self):
        df = self.dataPPP.df
        null_val = df.isnull().sum()
        # null_val = df.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
        percent = 100 * null_val/len(df)
        # list_type = df.isnull().sum().dtypes #データ型
        Missing_table = pd.concat([null_val, percent], axis = 1)
        missing_table_len = Missing_table.rename(
        columns = {0:'欠損値', 1:'%', 2:'type'})
        return missing_table_len.sort_values(by=['欠損値'], ascending=False)

