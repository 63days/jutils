import matplotlib.pyplot as plt
import jutils
import numpy as np
import torch
from typing import Sequence, Union


def normalize_points(p, method: str="sphere"):
    if method == "sphere":
        return _to_unit_sphere(p)
    elif method == "cube":
        return _to_unit_cube(p)
    else:
        raise AssertionError

def _to_unit_sphere(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    dtype = type(pc)
    pc = jutils.nputil.np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1, N, 3)
    m = pc.mean(1, keepdim=True)
    pc = pc - m
    s = torch.max(torch.sqrt(torch.sum(pc**2, -1, keepdim=True)), 1, keepdim=True)[0]
    pc = pc / s
    pc = pc.reshape(shapes)
    if dtype == np.ndarray:
        return jutils.thutil.th2np(pc)
    return pc

def _to_unit_cube(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    dtype = type(pc)
    pc = jutils.nputil.np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1,N,3)
    max_vals = pc.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = pc.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]
    
    pc = pc - center
    pc = pc / max_range[..., None]
    pc = pc.reshape(shapes)    
    if dtype == np.ndarray:
        return jutils.thutil.th2np(pc)
    return pc

def plot_pc(pointclouds: np.ndarray, save_path=None):
    """
    pointclouds: [H,W,N,3]
    """
    if isinstance(pointclouds, torch.Tensor):
        pointclouds = pointclouds.clone().detach().cpu().numpy()
    assert isinstance(pointclouds, np.ndarray)

    if len(pointclouds.shape) == 2:
        pointclouds = pointclouds[None, None, :]
    elif len(pointclouds.shape) == 3:
        pointclouds = pointclouds[None, :]

    assert len(pointclouds.shape) == 4
    print(pointclouds.shape)
    H, W, N, _ = pointclouds.shape
    fig, axs = plt.subplots(
            nrows=H, ncols=W, figsize=(W * 3.4, H * 3.4), squeeze=False, subplot_kw={"projection": "3d"}
    )

    for i in range(H):
        for j in range(W):
            pc = pointclouds[i, j]
            ax = axs[i][j]
            ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1])
            ax.axis("off")
    plt.subplots_adjust(hspace=0)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
