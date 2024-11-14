import numpy as np
import scipy.stats

def SE(pred, target):
    return (pred-target)**2

def RMSE(pred, target):
    return np.sqrt(np.mean(SE(pred, target)))

def AE(pred, target):
    return np.abs(pred-target)

def MAE(pred, target):
    return np.mean(AE(pred, target))

def SF(pred, target, reduce=True):
    s = np.exp(-(pred-target)/10) - 1
    s[pred >= target] = (np.exp((pred-target)/13) - 1)[pred >= target]
    return np.sum(s) if reduce else s


def PICP(pred, target, alpha=0.95, reduce=True):
    s = np.zeros(shape=(target.shape[0]))
    #lows, ups = scipy.stats.truncnorm.interval(alpha, -pred[0]/pred[1], np.inf, loc=pred[0], scale=pred[1])
    lows, ups = scipy.stats.lognorm.interval(alpha, s=pred[1], scale=np.exp(pred[0]))
    s[np.logical_and(target >= lows, target <= ups)] = 1
    return np.mean(s) if reduce else s

def NMPIW(pred, target, alpha=0.95, reduce=True):
    #lows, ups = scipy.stats.truncnorm.interval(alpha, -pred[0]/pred[1], np.inf, loc=pred[0], scale=pred[1])
    lows, ups = scipy.stats.lognorm.interval(alpha, s=pred[1], scale=np.exp(pred[0]))
    return np.mean(ups-lows) if reduce else (ups-lows)
