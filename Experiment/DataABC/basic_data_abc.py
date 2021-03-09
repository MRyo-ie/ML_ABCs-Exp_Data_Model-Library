
from abc import ABCMeta, abstractmethod
from .abcs.abc_data import DataABC


class Raw_Data(DataABC):
    """
    データ配分を特に変更せず、そのまま出力する。
    """
    def get_train(self):
        # X = self.X_train
        # y = self.Q_train * self.Y_train + (1 - self.Q_train) * (1 - self.Y_train)
        return {
            'X': self.X_train,
            'Q': self.Q_train,
            'Y': self.Y_train,
        }

    def get_eval(self):
        return {
            'train' : {
                'X': self.X_train, 
                'Q': self.Q_train, 
                'Y': self.Y_train,
            },
            'valid' : { 
                'X': self.X_valid, 
                'Q': self.Q_valid, 
                'Y': self.Y_valid,
            },
            'test' : { 
                'X': self.X_test, 
                'Q': self.Q_test, 
                'Y': self.Y_test,
            }
        }


