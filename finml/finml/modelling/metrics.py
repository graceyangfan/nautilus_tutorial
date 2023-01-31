
import numpy as np 

class Metrics:
    NEG_LOG_LOSS = 'neg_log_loss'
    ACCURACY = 'accuracy'

def sharpe_ratio(returns: np.ndarray, axis=None):
    """
    Calculate the classic Sharpe ratio = mean(returns)/std(returns)
    """
    return np.mean(returns, axis=axis) / np.std(returns, axis=axis)