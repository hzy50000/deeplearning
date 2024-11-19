import numpy as np
import torch
import torch.nn.functional as F


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# 相关系数R
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def huber(true, pred, delta=1.0):
    true = torch.tensor(true, dtype=torch.float32)
    pred = torch.tensor(pred, dtype=torch.float32)
    loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                       delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
    return torch.mean(loss).item()


# log cosh 损失
def log_cosh_loss(pred, target):
    pred = torch.tensor(pred, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    loss = torch.mean(torch.log(torch.cosh(pred - target)))
    return loss


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    huber_loss = huber(pred, true, 1.0)
    log_cosh = log_cosh_loss(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, huber_loss, log_cosh
