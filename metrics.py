import numpy as np

class Metrics():

    def r2(y_true, y_pred):
        return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def rmse(y_true, y_pred):
        return np.sqrt(Metrics.mse(y_true, y_pred))
