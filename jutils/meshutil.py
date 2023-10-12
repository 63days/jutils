import trimesh
from enum import Enum
import torch
import numpy as np
from jutils import nputil, thutil
import jutils

def write_obj(name: str, vertices: np.ndarray, faces: np.ndarray):
    """
    name: filename
    vertices: (V,3)
    faces: (F,3) Assume the mesh is a triangle mesh.
    """
    vertices = thutil.th2np(vertices)
    faces = thutil.th2np(faces).astype(np.uint32)
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(faces)):
        fout.write("f "+str(faces[ii,0]+1)+" "+str(faces[ii,1]+1)+" "+str(faces[ii,2]+1)+"\n")
    fout.close()
    

def write_obj_triangle(name: str, vertices: np.ndarray, triangles: np.ndarray):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()


def write_obj_polygon(name: str, vertices: np.ndarray, polygons: np.ndarray):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii][0])+" "+str(vertices[ii][1])+" "+str(vertices[ii][2])+"\n")
    for ii in range(len(polygons)):
        fout.write("f")
        for jj in range(len(polygons[ii])):
            fout.write(" "+str(polygons[ii][jj]+1))
        fout.write("\n")
    fout.close()

def read_obj(name: str):
    verts = []
    faces = []
    with open(name, "r") as f:
        lines = [line.rstrip() for line in f]

        for line in lines:
            if line.startswith("v "):
                verts.append(np.float32(line.split()[1:4]))
            elif line.startswith("f "):
                faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

        v = np.vstack(verts)
        f = np.vstack(faces) - 1
        return v, f



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

class SampleBy(Enum):
    AREAS = 0
    FACES = 1
    HYB = 2

def get_faces_normals(mesh):
    if type(mesh) is not torch.Tensor:
        vs, faces = mesh
        vs_faces = vs[faces]
    else:
        vs_faces = mesh
    if vs_faces.shape[-1] == 2:
        vs_faces = torch.cat(
            (vs_faces, torch.zeros(*vs_faces.shape[:2], 1, dtype=vs_faces.dtype, device=vs_faces.device)), dim=2)
    face_normals = torch.cross(vs_faces[:, 1, :] - vs_faces[:, 0, :], vs_faces[:, 2, :] - vs_faces[:, 1, :])
    return face_normals

def compute_face_areas(mesh):
    face_normals = get_faces_normals(mesh)
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_areas_ = face_areas.clone()
    face_areas_[torch.eq(face_areas_, 0)] = 1
    face_normals = face_normals / face_areas_[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals

def sample_uvw(shape, device):
    u, v = torch.rand(*shape, device=device), torch.rand(*shape, device=device)
    mask = (u + v).gt(1)
    u[mask], v[mask] = -u[mask] + 1, -v[mask] + 1
    w = -u - v + 1
    uvw = torch.stack([u, v, w], dim=len(shape))
    return uvw

def sample_on_mesh(mesh, num_samples: int, face_areas = None,
                   sample_s = SampleBy.HYB):
    vs, faces = mesh
    if faces is None:  # sample from pc
        uvw = None
        if vs.shape[0] < num_samples:
            chosen_faces_inds = torch.arange(vs.shape[0])
        else:
            chosen_faces_inds = torch.argsort(torch.rand(vs.shape[0]))[:num_samples]
        samples = vs[chosen_faces_inds]
    else:
        weighted_p = []
        if sample_s == SampleBy.AREAS or sample_s == SampleBy.HYB:
            if face_areas is None:
                face_areas, _ = compute_face_areas(mesh)
            face_areas[torch.isnan(face_areas)] = 0
            weighted_p.append(face_areas / face_areas.sum())
        if sample_s == SampleBy.FACES or sample_s == SampleBy.HYB:
            weighted_p.append(torch.ones(mesh[1].shape[0], device=mesh[0].device))
        chosen_faces_inds = [torch.multinomial(weights, num_samples // len(weighted_p), replacement=True) for weights in weighted_p]
        if sample_s == SampleBy.HYB:
            chosen_faces_inds = torch.cat(chosen_faces_inds, dim=0)
        chosen_faces = faces[chosen_faces_inds]
        uvw = sample_uvw([num_samples], vs.device)
        samples = torch.einsum('sf,sfd->sd', uvw, vs[chosen_faces])
    return samples, chosen_faces_inds, uvw

def repair_normals(v, f):
    mesh = trimesh.Trimesh(v, f)
    trimesh.repair.fix_normals(mesh)
    v = mesh.vertices
    f = np.asarray(mesh.faces)
    return v, f


###### Teomporal ######
import matplotlib.pyplot as plt
def get_gm_support(gm, x, ignore_phi=False):
    dim = x.shape[-1]
    mu, p, phi, eigen = gm
    sigma_det = eigen.prod(-1)
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None]).squeeze(1)
    phi = torch.softmax(phi, dim=2)
    if ignore_phi:
        phi = torch.ones_like(phi)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    distance = x[:, :, None, :] - mu
    mahalanobis_distance = - .5 * torch.einsum('bngd,bgdc,bngc->bng', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
    mahalanobis_distance -= const_2[:, :, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    return support, const_2

def get_gm_euclidean_support(gm, x):
    dim = x.shape[-1]
    mu, p, phi, eigen = gm
    distance = x[:, :, None, :] - mu
    
    euc_distance = -(distance ** 2).sum(-1)

    const_2, _ = euc_distance.max(dim=2)
    euc_distance -= const_2[:, :, None]

    return euc_distance, const_2
    mahalanobis_distance = - .5 * torch.einsum('bngd,bgdc,bngc->bng', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
    mahalanobis_distance -= const_2[:, :, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    return support, const_2


def gm_log_likelihood_loss(gms, x, get_supports: bool = False,
                           mask = None, reduction: str = "mean", ignore_phi=False):

    batch_size, num_points, dim = x.shape
    support, const = get_gm_euclidean_support(gms, x)
    probs = torch.log(support.sum(dim=2)) + const
    if mask is not None:
        probs = probs.masked_select(mask=mask.flatten())
    if reduction == 'none':
        likelihood = probs.sum(-1)
        loss = - likelihood / num_points
    else:
        likelihood = probs.sum()
        loss = - likelihood / (probs.shape[0] * probs.shape[1])
    if get_supports:
        return loss, support
    return loss


def split_mesh_by_gmm(mesh, gmm):
    faces_split = {}
    vs, faces = mesh
    vs_mid_faces = vs[faces].mean(1)
    _, supports = gm_log_likelihood_loss(gmm, vs_mid_faces.unsqueeze(0), get_supports=True)
    supports = supports[0]
    label = supports.argmax(1)
    for i in range(gmm[1].shape[2]):
        select = label.eq(i)
        if select.any():
            faces_split[i] = faces[select]
        else:
            faces_split[i] = None
    return faces_split


def flatten_gmm(gmm):
    b, gp, g, _ = gmm[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmm]
    p = p.reshape(*p.shape[:2], -1)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2)
    return z_gmm


def flatten_gmms_item(x):
    """
    Input: [B,1,G,*shapes]
    Output: [B,G,-1]
    """
    return x.reshape(x.shape[0], x.shape[2], -1)

@torch.no_grad()
def batch_gmms_to_gaus(gmms):
    """
    Input:
        [T(B,1,G,3), T(B,1,G,3,3), T(B,1,G), T(B,1,G,3)]
    Output:
        T(B,G,16)
    """
    if isinstance(gmms[0], list):
        gaus = gmms[0].copy()
    else:
        gaus = list(gmms).copy()
    
    gaus = [flatten_gmms_item(x) for x in gaus]
    return torch.cat(gaus, -1)

@torch.no_grad()
def batch_gaus_to_gmms(gaus, device="cpu"):
    """
    Input: T(B,G,16)
    Output: [mu: T(B,1,G,3), eivec: T(B,1,G,3,3), pi: T(B,1,G), eival: T(B,1,G,3)]
    """
    gaus = jutils.nputil.np2th(gaus).to(device)
    if len(gaus.shape) < 3:
        gaus = gaus.unsqueeze(0) # expand dim for batch

    B,G,_ = gaus.shape
    mu = gaus[:,:,:3].reshape(B,1,G,3)
    eivec = gaus[:,:,3:12].reshape(B,1,G,3,3)
    pi = gaus[:,:,12].reshape(B,1,G)
    eival = gaus[:,:,13:16].reshape(B,1,G,3)
    
    return [mu, eivec, pi, eival]

def get_vertex_color_from_gaus(vs, gaus, ignore_phi=True):
    gmm = batch_gaus_to_gmms(gaus)
    vs = nputil.np2th(vs)
    if vs.ndim == 2:
        vs = vs.unsqueeze(0)

    _, supports = gm_log_likelihood_loss(gmm, vs, get_supports=True, ignore_phi=ignore_phi)
    # supports = get_gm_support(gmm, vs, ignore_phi=True)
    supports = supports[0]
    label = supports.argmax(1)

    cmap = plt.get_cmap("turbo")
    func = lambda x : cmap(x / 16)[..., :3]
    return func(label)
