""" Compute standard cost-sensitive losses and metrics for models """
import torch

def rul_score(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    y_diff = y_pred - y_true
    return torch.sum(torch.where(y_diff < 0, torch.exp(-y_diff/13)-1, torch.exp(y_diff/10)-1))

def mae(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    return torch.mean(abs(y_true - y_pred))

def mse(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss()(y_pred, y_true)

def rmse(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.MSELoss()(y_pred, y_true))

def rmse_window(y_true : torch.Tensor, y_pred : torch.Tensor, window_size : int):
    return torch.sqrt(torch.mean((y_true[-window_size] - y_pred[-window_size])**2))

def lin_lin(a : int, b : int) -> torch.Tensor:
    def lin_lin_loss(y_true : torch.Tensor, y_pred : torch.Tensor):
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, -a * error, b * error))
    return lin_lin_loss

def lin_se(a : int) -> torch.Tensor:
    def lin_se_loss(y_true : torch.Tensor, y_pred : torch.Tensor):
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, -a * error, torch.square(error)))
    return lin_se_loss

def sle_se(y_true : torch.Tensor, y_pred : torch.Tensor):
    return torch.mean(torch.where(y_pred < y_true, 
                                  torch.square(torch.log(y_pred+1) - torch.log(y_true+1)), 
                                  torch.square(y_pred - y_true)
                                  ))

def quad_quad(a : float) -> torch.Tensor:
    def quad_quad_loss(y_true : torch.Tensor, y_pred : torch.Tensor):
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, 2*a*torch.square(error), 2*(-a+1)*torch.square(error)))
    return quad_quad_loss

def custom_loss_threshold_overestimating(t1 : float, t2 : float) -> torch.Tensor:        
    def custom_loss(y_true : torch.Tensor, y_pred : torch.Tensor):
        loss = torch.where((t1 < y_true) & (y_true <= t2), 
                                       abs(y_true - y_pred), y_true * (y_true - y_pred))
        loss += torch.where(y_pred > y_true, 
                           abs(1/y_true * (y_true - y_pred), 0))
        return loss.mean()
    return custom_loss

def custom_loss_threshold_no_estimation(t1 : float, t2 : float) -> torch.Tensor:
    def custom_loss(y_true : torch.Tensor, y_pred : torch.Tensor):
        return torch.where((t1 < y_true) & (y_true <= t2), 
                           abs(y_true - y_pred), 0).mean()
                                
    return custom_loss