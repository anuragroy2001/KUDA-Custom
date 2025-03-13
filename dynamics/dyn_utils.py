import os
import numpy as np
import cv2
import torch
import PIL.Image as Image
import moviepy.editor as mpy
from dynamics.mlp.mlp import MLP
import open3d as o3d

from utils import farthest_point_sampling, load_config


# material specific, control the loading of checkpoint and config
def load_config_and_ckpt(material):
    if material == 'rope':
        config_path = './configs/rope.yaml'
        ckpt_path = 'dynamics/checkpoints/rope_40_02_06_fri03.pth'
    elif material == 'cube':
        config_path = './configs/cube.yaml'
        ckpt_path = 'dynamics/checkpoints/cube.pth'
    elif material == 'granular':
        config_path = './configs/granular.yaml'
        ckpt_path = 'dynamics/checkpoints/granular_02_06.pth'
    elif material == 'T_shape':
        config_path = './configs/pushing_T_mlp.yaml'
        ckpt_path = 'dynamics/checkpoints/pushing_T_mlp.pth'
    else:
        raise ValueError('Unknown material')
    config = load_config(config_path)
    ckpt = torch.load(ckpt_path)
    return config, ckpt

def pad(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = np.zeros((max_dim, x.shape[1]), dtype=np.float32)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = np.zeros((x.shape[0], max_dim, x.shape[2]), dtype=np.float32)
        x_pad[:, :x_dim] = x
    return x_pad

def pad_torch(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = torch.zeros((max_dim, x.shape[1]), dtype=x.dtype, device=x.device)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = torch.zeros((x.shape[0], max_dim, x.shape[2]), dtype=x.dtype, device=x.device)
        x_pad[:, :x_dim] = x
    return x_pad

def quaternion_to_rotation_matrix(q):
    # Extract the values from q
    q1, q2, q3, w = q
    
    # First row of the rotation matrix
    r00 = 1 - 2 * (q2 ** 2 + q3 ** 2)
    r01 = 2 * (q1 * q2 - q3 * w)
    r02 = 2 * (q1 * q3 + q2 * w)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q3 * w)
    r11 = 1 - 2 * (q1 ** 2 + q3 ** 2)
    r12 = 2 * (q2 * q3 - q1 * w)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q2 * w)
    r21 = 2 * (q2 * q3 + q1 * w)
    r22 = 1 - 2 * (q1 ** 2 + q2 ** 2)
    
    # Combine all rows into a single matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])
    
    return rotation_matrix

def rgb_colormap(repeat=1):
    base = np.asarray([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ])
    return np.repeat(base, repeat, axis=0)

def vis_points(points, intr, extr, img, point_size=3, point_color=(0, 0, 255)):
    # transform points
    point_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    point_homo = point_homo @ extr.T 
    
    point_homo[:, 1] *= -1
    point_homo[:, 2] *= -1
    
    # project points
    fx, fy, cx, cy = intr
    point_proj = np.zeros((point_homo.shape[0], 2))
    point_proj[:, 0] = point_homo[:, 0] * fx / point_homo[:, 2] + cx
    point_proj[:, 1] = point_homo[:, 1] * fy / point_homo[:, 2] + cy
    
    # visualize
    for k in range(point_proj.shape[0]):
        cv2.circle(img, (int(point_proj[k, 0]), int(point_proj[k, 1])), point_size,
                   point_color, -1)
    
    return point_proj, img

def opencv_merge_video(image_path, image_type, out_path, fps=20):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        if f'_{image_type}.jpg' in f_name:
            image_names.append(f_name)

    image_names.sort(key=lambda x: int(x.split('_')[0]))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(out_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    # print("Video merged!")

    video_writer.release()

def moviepy_merge_video(image_path, image_type, out_path, fps=20):
    # load images
    # f_names = os.listdir(image_path)
    # image_names = []
    # for f_name in f_names:
    #     if f'_{image_type}.jpg' in f_name:
    #         image_names.append(f_name)
    # image_names.sort(key=lambda x: int(x.split('_')[0]))
    
    # load images
    image_files = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(f'{image_type}.jpg')])
    
    # create a video clip from the images
    clip = mpy.ImageSequenceClip(image_files, fps=fps)
    # write the video clip to a file
    clip.write_videofile(out_path, fps=fps)

# in simulation
def construct_edges_from_states(states, adj_thresh, mask, tool_mask, no_self_edge=False, pushing_direction=None):  # helper function for construct_graph
    '''
    # :param states: (B, N+2M, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N+2M) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N+2M) torch tensor, true when index is a valid tool particle
    # :param pushing_direction: (B, 3) torch tensor, pushing direction for each eef particle
    
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    '''
    no_self_edge = False

    B, N, state_dim = states.shape
    # print(f'states shape: {states.shape}') # (64, 300, 3)
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)
    # print(f's_receiv shape: {s_receiv.shape}; s_sender shape: {s_sender.shape}') # (64, 300, 300, 3)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    # convert threshold to tensor
    threshold = torch.tensor(threshold, device=states.device, dtype=states.dtype)  # (B, )
    
    s_diff = s_receiv - s_sender # (B, N, N, 3)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # particle receiver
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # particle sender
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)  # tool particle receiver
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)  # tool particle sender
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations
    
    adj_matrix = ((dis - threshold[:, None, None]) < 0).to(torch.float32) # (B, N, N)

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    # Check before experiment
    # sugguest 5 for rope, 5 for cubes, 20 for coffee_beans
    topk = 10 #TODO: hyperparameter
    if topk > dis.shape[-1]:
        topk = dis.shape[-1]
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    
    n_rels = adj_matrix.sum(dim=(1,2))
    # print(f'n_rels: {n_rels}') # (64)
    # print(n_rels.shape, (mask * 1.0).sum(-1).mean().item(), n_rels.mean().item())
    n_rel = n_rels.max().long().item()
    # print('relation num:', n_rel)
    
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    # print(f'Rr shape: {Rr.shape}; Rs shape: {Rs.shape}') # (64, 410, 300); (64, 410, 300)
    # print(Rr, Rs)
    
    return Rr, Rs

def truncate_graph(data):
    Rr = data['Rr']
    Rs = data['Rs']
    Rr_nonzero = torch.sum(Rr, dim=-1) > 0
    Rs_nonzero = torch.sum(Rs, dim=-1) > 0
    n_Rr = torch.max(Rr_nonzero.sum(1), dim=0)[0].item()
    n_Rs = torch.max(Rs_nonzero.sum(1), dim=0)[0].item()
    max_n = max(n_Rr, n_Rs)
    data['Rr'] = data['Rr'][:, :max_n, :]
    data['Rs'] = data['Rs'][:, :max_n, :]
    return data
