
import numpy as np 
import torch 

class Metrics:
    NEG_LOG_LOSS = 'neg_log_loss'
    ACCURACY = 'accuracy'

def negtivate_MSE(pred, label):
    if isinstance(pred,torch.Tensor):
        loss = (pred - label) ** 2
        return -torch.mean(loss)
    elif isinstance(pred,np.ndarray):
        loss = (pred - label) ** 2
        return -np.mean(loss)

def sharpe_ratio(returns: np.ndarray, axis=None):
    """
    Calculate the classic Sharpe ratio = mean(returns)/std(returns)
    """
    return np.mean(returns, axis=axis) / np.std(returns, axis=axis)