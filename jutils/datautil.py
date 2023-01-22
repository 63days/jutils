import torch
import numpy as np

def th2np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError("Input should be either np.ndarray or torch.Tensor")

def np2th(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu().numpy()
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")

