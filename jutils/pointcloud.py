import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Sequence


def normalize_points(p):
    if isinstance(p, np.ndarray):
        return _normalize_points_numpy(p)
    elif isinstance(p, torch.Tensor):
        return _normalize_points_tensor(p)
    else:
        raise ValueError(f"type(p) is invalid.")


def _normalize_points_numpy(p: np.ndarray):
    shapes = p.shape
    N = shapes[-2]
    p = p.reshape(-1, N, 3)

    m = p.mean(1, keepdims=True)
    p = p - m
    s = np.max(np.sqrt(np.sum(p**2, -1, keepdims=True)), 1, keepdims=True)
    p = p / s

    p = p.reshape(shapes)
    return p


def _normalize_points_tensor(p: torch.Tensor):
    shapes = p.shape
    N = shapes[-2]
    p = p.reshape(-1, N, 3)
    m = p.mean(1, keepdim=True)
    p = p - m
    s = torch.max(torch.sqrt(torch.sum(p**2, -1, keepdim=True)), 1, keepdim=True)[0]
    p = p / s
    p = p.reshape(shapes)
    return p


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
