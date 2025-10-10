"""Compute standard cost-sensitive losses and metrics for models"""
import torch
import torch.nn as nn
from typing import Callable


class RULScore(nn.Module):
    """Remaining Useful Life Score loss function"""
    
    def __init__(self):
        super(RULScore, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_diff = y_pred - y_true
        return torch.sum(torch.where(y_diff < 0, torch.exp(-y_diff/13) - 1, torch.exp(y_diff/10) - 1))


class MAE(nn.Module):
    """Mean Absolute Error loss function"""
    
    def __init__(self):
        super(MAE, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y_pred - y_true))
    
class MSE(nn.Module):
    """Root Mean Squared Error loss function"""
    
    def __init__(self):
        super(MSE, self).__init__()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true)**2)


class RMSE(nn.Module):
    """Root Mean Squared Error loss function"""
    
    def __init__(self):
        super(RMSE, self).__init__()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_pred - y_true)**2))


class RMSEWindow(nn.Module):
    """Root Mean Squared Error for a specific window"""
    
    def __init__(self, window_size: int):
        super(RMSEWindow, self).__init__()
        self.window_size = window_size
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_pred[-self.window_size:] - y_true[-self.window_size:])**2))


class LinLin(nn.Module):
    """Linear-Linear asymmetric loss function"""
    
    def __init__(self, a: float, b: float):
        super(LinLin, self).__init__()
        self.a = a
        self.b = b
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, -self.a * error, self.b * error))


class LinSE(nn.Module):
    """Linear-Squared Error asymmetric loss function"""
    
    def __init__(self, a: float):
        super(LinSE, self).__init__()
        self.a = a
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, -self.a * error, error**2))


class SLESE(nn.Module):
    """Squared Log Error - Squared Error asymmetric loss function"""
    
    def __init__(self):
        super(SLESE, self).__init__()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure positive values for log
        y_pred_safe = torch.clamp(y_pred, min=1e-8)
        y_true_safe = torch.clamp(y_true, min=1e-8)
        
        log_error = (torch.log(y_pred_safe + 1) - torch.log(y_true_safe + 1))**2
        squared_error = (y_pred - y_true)**2
        
        return torch.mean(torch.where(y_pred < y_true, log_error, squared_error))


class QuadQuad(nn.Module):
    """Quadratic-Quadratic asymmetric loss function"""
    
    def __init__(self, a: float):
        super(QuadQuad, self).__init__()
        self.a = a
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        return torch.mean(torch.where(error < 0, 
                                    2 * self.a * error**2, 
                                    2 * (1 - self.a) * error**2))


class ThresholdOverestimating(nn.Module):
    """Custom threshold-based loss with overestimation penalty"""
    
    def __init__(self, t1: float, t2: float):
        super(ThresholdOverestimating, self).__init__()
        self.t1 = t1
        self.t2 = t2
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Base loss within threshold range
        in_range = (self.t1 < y_true) & (y_true <= self.t2)
        out_range = ~in_range
        
        base_loss = torch.where(in_range, 
                               torch.abs(y_pred - y_true), 
                               y_true * torch.abs(y_pred - y_true))
        
        # Additional penalty for overestimation
        overestimation_penalty = torch.where(
            y_pred > y_true,
            torch.abs((y_pred - y_true) / torch.clamp(y_true, min=1e-8)),
            torch.zeros_like(y_pred)
        )
        
        total_loss = base_loss + overestimation_penalty
        return torch.mean(total_loss)


class ThresholdNoEstimation(nn.Module):
    """Custom threshold-based loss with no estimation outside range"""
    
    def __init__(self, t1: float, t2: float):
        super(ThresholdNoEstimation, self).__init__()
        self.t1 = t1
        self.t2 = t2
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        in_range = (self.t1 < y_true) & (y_true <= self.t2)
        loss = torch.where(in_range, torch.abs(y_pred - y_true), torch.zeros_like(y_pred))
        return torch.mean(loss)


# Convenience functions for backward compatibility
def rul_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """RUL Score function"""
    return RULScore()(y_pred, y_true)


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error function"""
    return MAE()(y_pred, y_true)


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error function"""
    return torch.mean((y_pred - y_true)**2)


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error function"""
    return RMSE()(y_pred, y_true)


def rmse_window(y_pred: torch.Tensor, y_true: torch.Tensor, window_size: int) -> torch.Tensor:
    """RMSE for window function"""
    return RMSEWindow(window_size)(y_pred, y_true)


def lin_lin(a: float, b: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Linear-Linear loss factory function"""
    loss_fn = LinLin(a, b)
    return lambda y_pred, y_true: loss_fn(y_pred, y_true)


def lin_se(a: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Linear-SE loss factory function"""
    loss_fn = LinSE(a)
    return lambda y_pred, y_true: loss_fn(y_pred, y_true)


def sle_se(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """SLE-SE loss function"""
    return SLESE()(y_pred, y_true)


def quad_quad(a: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Quadratic-Quadratic loss factory function"""
    loss_fn = QuadQuad(a)
    return lambda y_pred, y_true: loss_fn(y_pred, y_true)


def custom_loss_threshold_overestimating(t1: float, t2: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Threshold overestimating loss factory function"""
    loss_fn = ThresholdOverestimating(t1, t2)
    return lambda y_pred, y_true: loss_fn(y_pred, y_true)


def custom_loss_threshold_no_estimation(t1: float, t2: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Threshold no estimation loss factory function"""
    loss_fn = ThresholdNoEstimation(t1, t2)
    return lambda y_pred, y_true: loss_fn(y_pred, y_true)