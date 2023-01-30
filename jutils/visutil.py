import numpy as np
import trimesh
import matplotlib.pyplot as plt
import mcubes
import torch
from jutils import fresnelvis, nputil, thutil
from PIL import Image
from typing import List


def plot(pc, lim=0.7, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if not isinstance(pc, list):
        pc = [pc]
    for p in pc:
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    return fig


def make_grid(bb_min=[-1, -1, -1], bb_max=[1, 1, 1], shapes=[64, 64, 64], flatten=True):
    coords = []
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shapes) is int:
        shape = np.array([shape] * bb_min.shape[0])
    for i, si in enumerate(shapes):
        coord = np.linspace(bb_min[i], bb_max[i], si)
        coords.append(coord)
    grid = np.stack(np.meshgrid(*coords, sparse=False), axis=-1)
    if flatten:
        grid = grid.reshape(-1, grid.shape[-1])
    return grid


def grid2mesh(grid, thresh=0, smooth=False, bbmin=-1, bbmax=1):
    if smooth:
        grid = mcubes.smooth(grid)

    verts, faces = mcubes.marching_cubes(grid, thresh)
    # verts = verts[:,[2,0,1]]
    verts = verts / (grid.shape[0] - 1)
    verts = verts * (bbmax - bbmin) + bbmin
    faces = faces.astype(int)
    return verts, faces


def render_grid(
    grid,
    thresh=0.0,
    shapes=(64, 64, 64),
    camera_kwargs=dict(
        camPos=np.array([2, 2, -2]),
        camLookat=np.array([0.0, 0.0, 0.0]),
        camUp=np.array([1, 0, 0]),
        camHeight=2,
        resolution=(512, 512),
        samples=16,
    ),
):
    grid = thutil.th2np(grid)
    grid = grid.reshape(shapes)
    verts, faces = grid2mesh(grid, thresh)
    img = fresnelvis.renderMeshCloud(
        mesh={"vert": verts, "face": faces}, **camera_kwargs
    )
    img = Image.fromarray(img)
    return img


def plot_pointcloud(pointcloud, color=None):
    pointcloud = thutil.th2np(pointcloud)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # lim = 0.7
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)

    ax.scatter(
        pointcloud[:, 0],
        pointcloud[:, 2],
        pointcloud[:, 1],
        c=color if color is not None else None,
    )

    ax.axis("off")
    return fig


def render_pointcloud(
    pointcloud,
    camPos=np.array([2, 2, -2]),
    camLookat=np.array([0.0, 0.0, 0.0]),
    camUp=np.array([1, 0, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
    cloudR=0.006,
):
    pointcloud = thutil.th2np(pointcloud)
    img = fresnelvis.renderMeshCloud(cloud=pointcloud, camPos=camPos, camLookat=camLookat, camUp=camUp, camHeight=camHeight, resolution=resolution, samples=samples, cloudR=cloudR)
    return Image.fromarray(img)


def plot_gaussians(
    gaussians, is_bspnet, multiplier=1.0, gaussians_colors=None, pc=None, pc_color=None
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    lim = 0.7
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    N = gaussians.shape[0]
    cmap = plt.get_cmap("jet")
    for i, g in enumerate(gaussians):
        if is_bspnet:
            mu, eival, eivec = g[:3], g[3:6], g[6:15]
        else:
            mu, eivec, eival = g[:3], g[3:12], g[13:]
        R = eivec.reshape(3, 3).T
        # R[..., [2,0]] = R[..., [0,2]]
        # eival[..., [2,0]] = eival[..., [0,2]]
        a, b, c = multiplier * np.sqrt(eival)
        # a, b, c = multiplier * np.ones(3)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones_like(u), np.cos(v))
        coord = np.stack((x, y, z), axis=-1)
        coord = np.einsum("ij,uvj->uvi", R, coord)  # apply rot on all coord
        x = coord[..., 0] + mu[0]
        y = coord[..., 1] + mu[1]
        z = coord[..., 2] + mu[2]
        if gaussians_colors is not None:
            color = gaussians_colors[i]
        else:
            color = cmap(i / N)
        ax.plot_surface(x, z, y, rstride=4, cstride=4, color=color)
        ax.axis("off")

    if pc is not None:
        ax.scatter(
            pc[:, 0],
            pc[:, 2],
            pc[:, 1],
            alpha=0.3,
            c=pc_color if pc_color is not None else None,
        )
    return fig


def render_gaussians(
    gaussians,
    is_bspnet=False,
    multiplier=1.0,
    gaussians_colors=None,
    attn_map=None,
    camera_kwargs=None,
):
    gaussians = thutil.th2np(gaussians)
    N = gaussians.shape[0]
    cmap = plt.get_cmap("jet")

    if attn_map is not None:
        assert N == attn_map.shape[0]
        vmin, vmax = attn_map.min(), attn_map.max()
        if vmin == vmax:
            normalized_attn_map = np.zeros_like(attn_map)
        else:
            normalized_attn_map = (attn_map - vmin) / (vmax - vmin)

        cmap = plt.get_cmap("viridis")
    camera_kwargs = (
        camera_kwargs
        if camera_kwargs is not None
        else dict(
            camPos=np.array([-2, 2, -2]),
            camLookat=np.array([0.0, 0.0, 0.0]),
            camUp=np.array([0, 1, 0]),
            camHeight=2,
            resolution=(512, 512),
            samples=16,
        )
    )
    lights = "rembrandt"
    renderer = fresnelvis.FresnelRenderer(lights=lights, camera_kwargs=camera_kwargs)
    for i, g in enumerate(gaussians):
        if is_bspnet:
            mu, eival, eivec = g[:3], g[3:6], g[6:15]
        else:
            mu, eivec, eival = g[:3], g[3:12], g[13:]
        R = eivec.reshape(3, 3).T
        scale = multiplier * np.sqrt(eival)
        scale_transform = np.diag((*scale, 1))
        rigid_transform = np.hstack((R, mu.reshape(3, 1)))
        rigid_transform = np.vstack((rigid_transform, [0, 0, 0, 1]))
        sphere = trimesh.creation.icosphere()
        sphere.apply_transform(scale_transform)
        sphere.apply_transform(rigid_transform)
        if attn_map is None and gaussians_colors is None:
            color = np.array(cmap(i / N)[:3])
        elif attn_map is not None:
            color = np.array(cmap(normalized_attn_map[i])[:3])
        else:
            color = gaussians_colors[i]

        renderer.add_mesh(
            sphere.vertices, sphere.faces, color=color, outline_width=None
        )
    image = renderer.render()
    return Image.fromarray(image)
