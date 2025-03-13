import yaml
import os
import numpy as np
import torch
import random
from PIL import Image
import io
import base64
import open3d as o3d


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config_real(config_path=None):
    if config_path is None:
        config_path = './configs/real_config.yaml'
    # args = parse_args(config_path)
    config = load_config(config_path)
    # wrap dict such that we can access config through attribute
    class ConfigDict(dict):
        def __init__(self, config):
            """recursively build config"""
            self.config = config
            for key, value in config.items():
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]
        def __getstate__(self):
            return self.config
        def __setstate__(self, state):
            self.config = state
            self.__init__(state)
    config = ConfigDict(config)
    return config

def load_prompt(prompt_fname):
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # get full path to file
    if '/' in prompt_fname:
        prompt_fname = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *prompt_fname)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    # read file
    with open(full_path, 'r') as f:
        contents = f.read().strip()
    return contents

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def encode_image(image):
    if not isinstance(image, Image.Image):
        # must be array in bgr
        image = Image.fromarray(image[..., ::-1])

    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    return encoded_image

def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
    """voxelize a point cloud"""
    pc = pc.astype(np.float32)
    # make sure the point is within the voxel bounds
    pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
    # voxelize
    voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
    # to integer
    _out = np.empty_like(voxels)
    voxels = np.round(voxels, 0, _out).astype(np.int32)
    assert np.all(voxels >= 0)
    assert np.all(voxels < map_size)
    return voxels

def voxel2index(voxels, map_size):
    """convert 3D voxel coordinates to 1D index"""
    return voxels[:, 2] * map_size * map_size + voxels[:, 1] * map_size + voxels[:, 0]

def farthest_point_sampling(points, num_points, start_idx=None, return_idx=False):
    """
    Farthest point sampling algorithm to sample 'num_points' points from the input points.
    """
    points = np.array(points, dtype=np.float32)
    if len(points) <= num_points:
        if return_idx:
            return np.arange(len(points))
        else:
            return points

    sampled_indices = []
    if start_idx is None:
        # use the center point as the starting point
        points = np.concatenate([points.mean(axis=0, keepdims=True), points], axis=0)
        sampled_indices.append(0)
    else:
        sampled_indices.append(start_idx)
        num_points -= 1
    distances = np.linalg.norm(points - points[sampled_indices[0]], axis=1)

    for _ in range(num_points):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        new_distances = np.linalg.norm(points - points[sampled_indices[-1]], axis=1)
        distances = np.minimum(distances, new_distances)
    
    if start_idx is None:
        if return_idx:
            return np.array(sampled_indices[1:], dtype=int) - 1
        else:
            return points[sampled_indices[1:]]
    else:
        if return_idx:
            return sampled_indices
        else:
            return points[sampled_indices]

def fps_rad_idx(pcd, radius):
    # pcd: (n, dim) numpy array
    # pcd_fps: (-1, dim) numpy array
    # radius: float
    # keep order in the result of fps
    pcd_fps_lst = [pcd[0]]
    idx_lst = [0]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        idx_lst.append(dist.argmax())
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    idx_lst = np.stack(idx_lst, axis=0)
    return pcd_fps, idx_lst

# to get state in real env
def truncate_points(obj_kps_list, fps_radius, visualize=False):
    # obj_kps_list: list of numpy array, each of shape (n, 3)
    max_nobj = 200

    if not isinstance(obj_kps_list, list):
        obj_kps_list = [obj_kps_list]

    fps_idx_list = []
    ## sampling using raw particles
    for j in range(len(obj_kps_list)):
        # farthest point sampling
        particle = obj_kps_list[j]
        if particle.ndim == 1:
            continue
        fps_idx_1 = farthest_point_sampling(particle, min(max_nobj, particle.shape[0]), return_idx=True)

        # downsample to uniform radius
        downsample_particle = particle[fps_idx_1, :]
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(int)
        fps_idx = fps_idx_1[fps_idx_2]
        fps_idx_list.append(fps_idx)

    obj_kps_list = [obj_kps_list[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_kps_list[0])
        pcd.paint_uniform_color([0, 1, 0])

        # visualize edges
        # edges = []
        # for i in range(Rr.shape[0]):
        #     edges.append([Rs[i].argmax(), Rr[i].argmax()])
        # edges = np.array(edges).astype(np.int32)
        # pcd_edges = o3d.geometry.LineSet()
        # pcd_edges.points = o3d.utility.Vector3dVector(state)
        # pcd_edges.lines = o3d.utility.Vector2iVector(edges)
        # pcd_edges.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(edges)))
        # visualize_o3d([pcd, pcd_eef, pcd_edges])

    return obj_kps_list
