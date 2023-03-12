from trimesh.visual.color import colorsys


try:
    import matplotlib.pyplot as plt
    from typing import Union, Tuple
    import mitsuba as mi
    from PIL import Image
    try:
        mi.set_variant("cuda_ad_rgb")
        print("cuda mitsuba")
    except:
        mi.set_variant("llvm_ad_rgb")  # DO NOT BREAKLINE
    from mitsuba import ScalarTransform4f as T
    import numpy as np
    import torch
    from pathlib import Path
    import multiprocessing as mp
    import trimesh
    from rich.progress import track
    import pickle
    import gc
    import drjit as dr
    import trimesh
    import jutils.meshutil
    
    gray_color = np.ones(3) * 0.45
    selec_color = np.array([0.65882353, 0.21960784, 0.19607843])
    render_colors = {
        "chair": np.array([96, 153, 102]) / 255,
        "airplane": np.array([176, 139, 187]) / 255,
    }
    part_colors = (
        np.array([[153, 50, 204], [0, 64, 255], [99, 255, 32], [250, 253, 15]]) / 255
    )

    def to_mitsuba_coord(pc):
        # rot_mat = torch.tensor([[1, 0, 0], [0,0,-1], [0,1,0]]).double()
        rot_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        return (pc * 2) @ rot_mat.T

    phis = np.array([165.07222439, 311.02125986, 9.83108249, 223.51034837, 76.80885784])

    def load_and_process_mesh(path):
        v, f = jutils.meshutil.read_obj(path)
        v, f = jutils.meshutil.repair_normals(v, f)
        v = jutils.pcutil.normalize_points(v, "cube")
        return v, f
    def clean_cache():
        gc.collect()
        dr.eval()
        dr.sync_thread()
        dr.flush_malloc_cache()
        dr.malloc_clear_statistics()

    def load_sensor(phi, r=10, theta=60.0):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f(
            [0, 0, r]
        )

        return mi.load_dict(
            {
                "type": "perspective",
                "fov": 39.3077,
                "to_world": T.look_at(origin=origin, target=[0, 0, 0], up=[0, 0, 1]),
                "sampler": {"type": "independent", "sample_count": 16},
                "film": {
                    "type": "hdrfilm",
                    "width": 256,
                    "height": 256,
                    "rfilter": {
                        "type": "tent",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    sensors = [load_sensor(phi) for phi in phis]

    def get_scene_dict(scene_type="default", floor=True):
        default_scene = {
            "type": "scene",
            "integrator": {"type": "path"},
            "light": {"type": "constant", "radiance": 1.0},
        }

        if scene_type == "default":
            out_scene = default_scene
        else:
            out_scene = default_scene

        if floor:
            out_scene["floor"] = {
                "type": "rectangle",
                "to_world": T.translate([0, 0, -2]).scale(100),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": 1.0},
                },
            }

        return out_scene

    def get_sensor(r, phi, theta, res):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f(
            [0, 0, r]
        )

        if type(res) == int:
            res = (res, res)

        return mi.load_dict(
            {
                "type": "perspective",
                "type": "perspective",
                "fov": 39.3077,
                "to_world": T.look_at(origin=origin, target=[0, 0, 0], up=[0, 0, 1]),
                "sampler": {"type": "independent", "sample_count": 16},
                "film": {
                    "type": "hdrfilm",
                    "width": res[0],
                    "height": res[1],
                    "rfilter": {
                        "type": "box",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    def render_pointcloud(
        pc,
        color=0.6,
        normalize=None,
        camR=10,
        camPhi=45,
        camTheta=60,
        resolution=(512, 512),
        **scene_kwargs,
    ):
        scene_dict = get_scene_dict(**scene_kwargs)
        pc = jutils.thutil.th2np(pc)
        if normalize is not None:
            pc = jutils.pcutil.normalize_points(pc, normalize)
        mit_pc = to_mitsuba_coord(pc) - np.array([-0.25, -0.25, 0])

        for i, pos in enumerate(mit_pc):
            scene_dict[f"point_{i}"] = {
                "type": "sphere",
                "to_world": T.translate(pos).scale(0.05),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": color},
                },
            }

        scene = mi.load_dict(scene_dict)
        sensor = get_sensor(camR, camPhi, camTheta, resolution)

        return mi.render(scene, spp=1000, sensor=sensor)
    
    def vf2mimesh(v, f, color):
        bsdf = mi.load_dict(
            {"type": "diffuse", "reflectance": {"type": "rgb", "value": color}}
        )
        props = mi.Properties()
        props["mesh_bsdf"] = bsdf
        mimesh = mi.Mesh(
            "mymesh",
            vertex_count=v.shape[0],
            face_count=f.shape[0],
            has_vertex_normals=False,
            has_vertex_texcoords=False,
            props=props,
        )
        mesh_params = mi.traverse(mimesh)
        mesh_params["vertex_positions"] = np.ravel(v)
        mesh_params["faces"] = np.ravel(f)
        # mesh_params["bsdf.reflectance.value"] = color
        mesh_params.update()
        return mimesh

    def vf2mimesh_vertex_color(v, f, color=0.45):
        vert_bsdf = mi.load_dict({
            "type": "diffuse",
            "reflectance": {
                "type": "mesh_attribute",
                "name": "vertex_color",
            },
        })
        props = mi.Properties()
        props["vertex_bsdf"] = vert_bsdf


        mimesh = mi.Mesh("mymesh",
            vertex_count=v.shape[0],
            face_count=f.shape[0],
            has_vertex_normals=False,
            has_vertex_texcoords=False,
            props=props
        )
        mimesh.add_attribute("vertex_color", 3,
                             [0] * (v.shape[0] * 3))
        
        if isinstance(color, float):
            mask_color = np.ones((v.shape[0], 3)) * color
        elif isinstance(color, list):
            mask_color = np.array(mask_color)
        else:
            mask_color = color
        # mask_color = np.zeros((v.shape[0], 3))
        # mask_color[:] = color
        # mask_color[v_mask] = select_color

        mesh_params = mi.traverse(mimesh)
        mesh_params["vertex_positions"] = np.ravel(v)
        mesh_params["vertex_color"] = np.ravel(mask_color)
        mesh_params["faces"] = np.ravel(f)
        mesh_params.update()

        return mimesh

    def vf2bbox(v, f, color):
        bsdf = mi.load_dict({
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': 0.7,
            'int_ior': 'bk7',
            'ext_ior': 'air',
            'specular_reflectance': {'type': 'rgb', 'value': color}
            })
        props = mi.Properties()
        props["mesh_bsdf"] = bsdf
        mimesh = mi.Mesh(
            "mybbox",
            vertex_count=v.shape[0],
            face_count=f.shape[0],
            has_vertex_normals=False,
            has_vertex_texcoords=False,
            props=props
        )
        mesh_params = mi.traverse(mimesh)
        mesh_params["vertex_positions"] = np.ravel(v)
        mesh_params["faces"] = np.ravel(f)
        mesh_params.update()

        return mimesh

    def render_mesh_vertex_color(
        v: Union[np.ndarray, torch.Tensor],
        f: Union[np.ndarray, torch.Tensor],
        color: Union[int, np.ndarray] = 0.45,
        bbox: Tuple =None,
        bbox_color=np.array([0.53, 0.68, 0.92]),
        normalize=None,
        camR=10,
        camPhi=45,
        camTheta=60,
        resolution=(512, 512),
        floor=True,
        **scene_kwargs,
        ):
        clean_cache()
        scene_dict = get_scene_dict(**scene_kwargs, floor=floor)

        v = jutils.thutil.th2np(v)
        f = jutils.thutil.th2np(f)
        if normalize is not None:
            v = jutils.pcutil.normalize_points(v, normalize)
        v = to_mitsuba_coord(v)
        # scene_dict["mesh"] = vf2mimesh(v,f, color)
        scene_dict["mesh"] = vf2mimesh_vertex_color(v, f, color)

        """
        bbox 
        """
        if bbox is not None:
            try:
                bv, bf = jutils.meshutil.read_obj("/home/juil/docker_home/salad/primitive_objs/cube.obj")
            except:
                bv, bf = jutils.meshutil.read_obj("/home/juil/salad/primitive_objs/cube.obj")
        
            minbb, maxbb = bbox
            transbb = (maxbb + minbb) / 2
            scalebb = (maxbb - minbb) / 2
            bv = (bv * scalebb) + transbb
            bv = to_mitsuba_coord(bv)
            scene_dict['bbox'] = vf2bbox(bv, bf, bbox_color)

        scene = mi.load_dict(scene_dict)
        sensor = get_sensor(camR, camPhi, camTheta, resolution)

        render = mi.render(scene, spp=1000, sensor=sensor)
        return render


    def render_mesh(
        v: Union[np.ndarray, torch.Tensor],
        f: Union[np.ndarray, torch.Tensor],
        color: Union[int, np.ndarray] = 0.45,
        bbox: Tuple =None,
        bbox_color=np.array([0.53, 0.68, 0.92]),
        normalize=None,
        camR=10,
        camPhi=45,
        camTheta=60,
        resolution=(512, 512),
        floor=True,
        **scene_kwargs,
    ):
        clean_cache()
        scene_dict = get_scene_dict(**scene_kwargs, floor=floor)

        v = jutils.thutil.th2np(v)
        f = jutils.thutil.th2np(f)
        if normalize is not None:
            v = jutils.pcutil.normalize_points(v, normalize)
        v = to_mitsuba_coord(v)
        scene_dict["mesh"] = vf2mimesh(v,f, color)

        """
        bbox 
        """
        if bbox is not None:
            try:
                bv, bf = jutils.meshutil.read_obj("/home/juil/docker_home/salad/primitive_objs/cube.obj")
            except:
                bv, bf = jutils.meshutil.read_obj("/home/juil/salad/primitive_objs/cube.obj")
        
            minbb, maxbb = bbox
            transbb = (maxbb + minbb) / 2
            scalebb = (maxbb - minbb) / 2
            bv = (bv * scalebb) + transbb
            bv = to_mitsuba_coord(bv)
            scene_dict['bbox'] = vf2bbox(bv, bf, bbox_color)

        scene = mi.load_dict(scene_dict)
        sensor = get_sensor(camR, camPhi, camTheta, resolution)

        render = mi.render(scene, spp=1000, sensor=sensor)
        return render
        img = Image.fromarray(np.array(render))
        return img

    def write_img(img, path):
        mi.util.write_bitmap(str(path), img)

    def load_and_process_mesh(path):
        v, f = jutils.meshutil.read_obj(path)
        v, f = jutils.meshutil.repair_normals(v, f )
        v = jutils.pcutil.normalize_points(v, "cube")
        return v, f
    def render_gaussians(gaussians, color=0.45, cmap=None, use_cmap=False,
                    transform=lambda x: x,
                    camR=10, camPhi=45, camTheta=60, camRes=(512,512), floor=True,
                    darken=False,
                    **scene_kwargs):
        clean_cache()
        scene_dict = get_scene_dict(**scene_kwargs, floor=floor)
        # cmap = plt.get_cmap("plasma")
        if isinstance(cmap, str):
            try:
                cmap = plt.get_cmap(cmap)
            except:
                pass
        else:
            cmap = plt.get_cmap("turbo")

        gaussians = jutils.thutil.th2np(gaussians)
        N = gaussians.shape[0]
        v_list = []
        f_list = []
        for i, g in enumerate(gaussians):
            mu, eivec, eival = g[:3], g[3:12], g[13:]

            R = eivec.reshape(3, 3).T
            S = np.sqrt(np.clip(eival, 1e-4, 99999)) * 0.45
        
            try:
                v, f = jutils.meshutil.read_obj("/home/juil/docker_home/salad/primitive_objs/sphere.obj")
            except:
                v, f = jutils.meshutil.read_obj("/home/juil/salad/primitive_objs/sphere.obj")
            
            v, f = jutils.meshutil.repair_normals(v, f)
            v = mu + ((v * S) @ R.T)
            v = to_mitsuba_coord(v) * 1.4
            mesh = trimesh.Trimesh(vertices=v, faces=f)
            mesh.fix_normals()
            v, f = np.array(mesh.vertices), np.array(mesh.faces)
            if use_cmap:
                c = np.array(cmap(i / N)[:3])
            else:
                if isinstance(color, (list, np.ndarray, torch.Tensor)):
                    c = color[i]
                else:
                    c = color
            if darken:
                c = darken_color(c)
            scene_dict[f'point_{i}'] = vf2mimesh(v, f, c)

        scene = mi.load_dict(scene_dict)
        sensor = get_sensor(camR, camPhi, camTheta, camRes)
        return mi.render(scene, spp=1000, sensor=sensor) 

    def darken_color(c):
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, min(1, l * 0.5), s=s)
except:
    pass
