"""Reusable task definitions."""
import numpy as np
import torch

__all__ = ['task_phi_position_fixed', 'task_phi_position_opt', 
           'task_phi_distance_fixed', 'task_phi_distance_opt']


def task_phi_position_fixed(shape: torch.Tensor, target_point: torch.Tensor, 
                          s_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error at fixed arc-length index."""
    x = shape[s_bar] - target_point
    return x, s_bar


def task_phi_position_opt(shape: torch.Tensor, target_point: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """Position error at closest point on shape."""
    distances = 0.5 * torch.sum((shape - target_point) ** 2, dim=1)
    _, s_star = torch.min(distances, dim=0)
    x = shape[s_star] - target_point
    return x, s_star


def task_phi_distance_fixed(shape: torch.Tensor, target_point: torch.Tensor, 
                          s_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance at fixed arc-length index."""
    x = 0.5 * torch.sum((shape[s_bar] - target_point) ** 2)
    return x, s_bar


def task_phi_distance_opt(shape: torch.Tensor, target_point: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal squared distance to shape."""
    distances = 0.5 * torch.sum((shape - target_point) ** 2, dim=1)
    x, s_star = torch.min(distances, dim=0)
    return x, s_star