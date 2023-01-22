import trimesh
import numpy as np
from jutils import nputil, thutil

def scene_as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values() if g.faces.shape[1] == 3))
    else:
        mesh = scene_or_mesh

    return mesh

def get_center(verts):
    max_vals = verts.max(0)
    min_vals = verts.min(0)
    center = (max_vals + min_vals) / 2
    return center

def to_center(verts):
    verts -= get_center(verts)[None, :]
    return verts

def get_offset_and_scale(verts, radius=1.):
    verts = thutil.th2np(verts)
    verts = verts.copy()
    
    offset = get_center(verts)[None,:]
    verts -= offset
    scale = 1 / np.linalg.norm(verts, axis=1).max() * radius
    
    return offset, scale

def normalize_mesh(mesh: trimesh.Trimesh):
    # unit cube normalization
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    maxv, minv = np.max(v, 0), np.min(v, 0)
    offset = minv
    v = v - offset
    scale = np.sqrt(np.sum((maxv - minv) ** 2))
    v = v / scale
    normed_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return dict(mesh=normed_mesh, offset=offset, scale=scale)



def normalize_scene(scene: trimesh.Scene):
    mesh_merged = scene_as_mesh(scene)

    out = normalize_mesh(mesh_merged)
    offset = out["offset"]
    scale = out["scale"]

    submesh_normalized_list = []
    for i, submesh in enumerate(list(scene.geometry.values())):
        v, f = np.array(submesh.vertices), np.array(submesh.faces)
        v = v - offset
        v = v / scale
        submesh_normalized_list.append(trimesh.Trimesh(v, f))
        
    return trimesh.Scene(submesh_normalized_list)
