
from .abcs.abc_model import ModelABC


class LSTM_model(ModelABC):
    """
    ・ LSTM
    """
    def __init__(self, model_instance=None):        
        super().__init__(model_instance)

    def fit(self, datadict):
        return super().fit(datadict)

    def predict(self, data):
        return super().predict(data)

    def predict_orgf(self, data):
        return super().predict_orgf(data)



class Attention_model(ModelABC):
    """
    ・ Attention
    """
    def __init__(self, model_instance=None):        
        super().__init__(model_instance)

    def fit(self, datadict):
        return super().fit(datadict)

    def predict(self, data):
        return super().predict(data)

    def predict_orgf(self, data):
        return super().predict_orgf(data)
