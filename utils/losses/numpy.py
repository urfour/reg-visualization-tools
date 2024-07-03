""" Compute standard cost-sensitive losses and metrics for models """
import numpy as np

def rul_score(error : np.ndarray) -> np.ndarray:
    return np.sum(np.where(error < 0, np.exp(-error/13)-1, np.exp(error/10)-1))

def ae(error : np.ndarray) -> np.ndarray:
    """ Absolute Error """
    return np.abs(error)

def se(error : np.ndarray) -> np.ndarray:
    """ Squared Error """
    return np.square(error)

def rse(error : np.ndarray) -> np.ndarray:
    """ Root-Squared Error """
    return np.sqrt(se(error))

def ape(y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
    """ Average Percentage Error """
    return np.abs(y_pred - y_true) / y_true

def rmse_window(error : np.ndarray, window_size : int):
    return np.sqrt(np.mean(rse(error[-window_size])))

def lin_lin(error : np.ndarray, a : int, b : int) -> np.ndarray:
    """ Asymmetric Linear - Linear Error """
    return np.where(error < 0, a * error, b * error)

def lin_lin_pe(y_pred : np.ndarray, y_true : np.ndarray, a : int, b : int) -> np.ndarray:
    """ Asymmetric Linear - Linear Percentage Error """
    return np.where(y_pred - y_true < 0, a * (y_pred - y_true)/y_true, b * (y_pred - y_true)/y_true)

def lin_lin2(error : np.ndarray, a : int) -> np.ndarray:
    """ Assymetric Linear - Linear Error only with alpha"""
    return np.where(error < 0, 2*a*error, 2*(1-a)*error)

def lin_se(error : np.ndarray, a : int) -> np.ndarray:
    """ Assymetric Linear - Squared Error"""
    return np.where(error < 0, a * error, np.square(error))

def abs_lin_lin(error : np.ndarray, a : int, b : int) -> np.ndarray:
    """ Asymmetric Absolute Linear - Linear Error"""
    return np.where(error < 0, -a * error, b * error)

def abs_lin_se(error : np.ndarray, a : int) -> np.ndarray:
    return np.where(error < 0, -a * error, np.square(error))

def sle_se(error : np.ndarray):
    return np.where(error < 0, 
                    np.square(np.log(-error+1)), 
                    np.square(error))

def log_lin(error : np.ndarray):
    return np.where(error < 0, np.log(error), error)

def lin_exp(error : np.ndarray):
    return np.where(error < 0, error, np.exp(error))

def quad_quad(error : np.ndarray, a : float) -> np.ndarray:
    return np.where(error < 0, 2*a*np.square(error), 2*(1-a)*np.square(error))