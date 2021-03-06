
from . import DataABC


class DemandForecasting_DataABC(DataABC):
    def __init__(self, dataPPP):
        super().__init__(dataPPP)
    
    def get_train(self):
        return super().get_train()

    def get_eval(self):
        return super().get_eval()

