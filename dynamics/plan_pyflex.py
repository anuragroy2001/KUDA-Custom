import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import cv2
import math
import os
import yaml
import glob
from functools import partial
import scipy
from copy import deepcopy

from envs.real_env import RealEnv
from dynamics.planner import Planner
from dynamics.dyn_utils import construct_edges_from_states, truncate_graph, pad, pad_torch, load_config_and_ckpt
from dynamics.gnn.model import DynamicsPredictor
from dynamics.tracker.spatracker_wrapper import SpaTrackerWrapper
from dynamics.video_recorder import VideoRecorder
from utils import set_seed, truncate_points

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

SIM_REAL_RATIO = 10.
PUSH_LENGTH = 0.1 # length in simulation

# ============ action format ============
def decode_action(action, push_length=0.2):
    x_start = action[..., 0]
    z_start = action[..., 1]
    theta = action[..., 2]
    length = action[..., 3]
    action_repeat = length.to(torch.int32)
    x_end = x_start - push_length * torch.cos(theta)
    z_end = z_start - push_length * torch.sin(theta)
    decoded_action = torch.stack([x_start, z_start, x_end, z_end], dim=-1)
    return decoded_action, action_repeat

# for closed-loop control
def step_and_record(env, action, video_recorder):
    # decode to full action
    x_start = action[0]
    z_start = action[1]
    theta = action[2]
    length = action[3]
    action_repeat = int(length)
    x_end = x_start - action_repeat * PUSH_LENGTH * math.cos(theta)
    z_end = z_start - action_repeat * PUSH_LENGTH * math.sin(theta)
    decoded_action = np.array([x_start, z_start, x_end, z_end])

    decoded_action = decoded_action_sim_to_real(decoded_action)
    x_start, y_start, x_end, y_end = decoded_action

    # rope
    # yaw = None
    # pusher
    yaw = -theta

    env.reset_to_default_pose()
    time.sleep(0.5)
    start_time = time.time()
    env.move_to_table_position(x_start, y_start, 0.1, yaw)
    env.move_to_table_position(x_start, y_start, 0.0, yaw)
    time.sleep(0.5)
    video_recorder.start()
    end_time = time.time()
    print("Move to start position took {:.4f} seconds".format(end_time - start_time))
    time.sleep(0.5)
    start_time = time.time()
    env.move_to_table_position(x_end, y_end, 0.0, yaw)
    env.move_to_table_position(x_end, y_end, 0.1, yaw)
    end_time = time.time()
    print("Move to end position took {:.4f} seconds".format(end_time - start_time))
    time.sleep(0.5)
    start_time = time.time()
    env.reset_to_default_pose()
    end_time = time.time()
    print("Reset to default pose took {:.4f} seconds".format(end_time - start_time))
    if video_recorder.is_alive():
        video_recorder.terminate()
        video_recorder.join()


# ============ coordinate transformation ============
def decoded_action_real_to_sim(action):
    """turn real action into simulator `particle` coordinate"""
    action_sim = deepcopy(action)
    action_sim[..., 1] *= -1
    action_sim[..., 3] *= -1
    action_sim *= SIM_REAL_RATIO
    return action_sim

def decoded_action_sim_to_real(action_sim):
    """turn action in simulator `particle` coordinate into real"""
    action_real = deepcopy(action_sim)
    action_real[..., 1] *= -1
    action_real[..., 3] *= -1
    action_real /= SIM_REAL_RATIO
    return action_real

def particle_real_to_sim(particles):
    if isinstance(particles, list):
        particles = np.array(particles).astype(np.float32)
    particles_sim = particles * SIM_REAL_RATIO
    particles_sim = particles_sim[:, [0, 2, 1]]  # (x, y, z) -> (x, z, y)
    particles_sim[:, 2] *= -1  # (x, z, y) -> (x, z, -y)
    return particles_sim

def particle_sim_to_real(particles_sim):
    if isinstance(particles_sim, list):
        particles_sim = np.array(particles_sim).astype(np.float32)
    particles_real = particles_sim / SIM_REAL_RATIO
    particles_real[:, 2] *= -1  # (x, z, -y) -> (x, z, y)
    particles_real = particles_real[:, [0, 2, 1]]  # (x, z, y) -> (x, y, z)
    return particles_real

# ============ distance metrics ============
def chamfer(x, y):  # x: (B, N, D), y: (B, M, D)
    n_points = 1000
    x = x[:, :n_points]  # (B, N, D)
    y = y[:, :n_points]  # (B, M, D)
    
    # Expand dimensions for broadcasting instead of using repeat
    x_expanded = np.expand_dims(x, axis=2)  # (B, N, 1, D)
    y_expanded = np.expand_dims(y, axis=1)  # (B, 1, M, D)
    
    # Calculate pairwise distances
    dis = np.linalg.norm(x_expanded - y_expanded, axis=-1)  # (B, N, M)
    
    # Calculate the Chamfer distance
    dis_xy = dis.min(axis=2).mean(axis=1)  # Mean over M
    dis_yx = dis.min(axis=1).mean(axis=1)  # Mean over N
    
    return dis_xy + dis_yx

def em_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
    y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
    x_list = []
    y_list = []
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            print("Error in linear sum assignment!")
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2), dim=1)
    return emd

def end_point_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    # align the y(2th dim)-most point in x and y
    x_max = x[np.arange(x.size(0)), x[:, :, 1].argmax(dim=1)]
    x_min = x[np.arange(x.size(0)), x[:, :, 1].argmin(dim=1)]
    y_max = y[np.arange(y.size(0)), y[:, :, 1].argmax(dim=1)]
    y_min = y[np.arange(y.size(0)), y[:, :, 1].argmin(dim=1)]
    dis_max = torch.norm(torch.add(x_max, -y_max), 2, dim=1)
    dis_min = torch.norm(torch.add(x_min, -y_min), 2, dim=1)
    return dis_max + dis_min

def specification_square_distance(x, target_specification):
    # x: [B, N, D]
    # target_specification: ([n], [n, D])
    indices, target = target_specification
    x_target = x[:, indices]
    return torch.mean(torch.norm(torch.add(x_target, -target), 2, dim=2) ** 2, dim=1)

# ============ action sampling ============
def sample_action_seq(act_seq, action_lower_lim, action_upper_lim, n_sample, device, iter_index=0, noise_level=0.3):
    if iter_index == 0:
        # resample completely
        act_seqs = torch.rand((n_sample, act_seq.shape[0], act_seq.shape[1]), device=device) * \
            (action_upper_lim - action_lower_lim) + action_lower_lim
    else:
        # beta_filter = 0.7
        n_look_ahead = act_seq.shape[0]
        
        assert act_seq.shape[-1] == 4  # (x, y, theta, length)
        act_seqs = torch.stack([act_seq.clone()] * n_sample)
        xs = act_seqs[:, :, 0]
        ys = act_seqs[:, :, 1]
        thetas = act_seqs[:, :, 2]
        lengths = act_seqs[:, :, 3]
        
        x_ends = xs - lengths * PUSH_LENGTH * torch.cos(thetas)
        y_ends = ys - lengths * PUSH_LENGTH * torch.sin(thetas)

        # act_residual = torch.zeros((n_sample, 4), dtype=act_seqs.dtype, device=device)
        for i in range(n_look_ahead):
            noise_sample = torch.normal(0, noise_level, (n_sample, 4), device=device)
            beta = 0.1 * (10 ** i)
            act_residual = beta * noise_sample
            # act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)
            
            xs_i = xs[:, i] + act_residual[:, 0]
            ys_i = ys[:, i] + act_residual[:, 1]
            x_ends_i = x_ends[:, i] + act_residual[:, 2]
            y_ends_i = y_ends[:, i] + act_residual[:, 3]

            thetas_i = torch.atan2(ys_i - y_ends_i, xs_i - x_ends_i)
            lengths_i = torch.norm(torch.stack([x_ends_i - xs_i, y_ends_i - ys_i], dim=-1), dim=-1).clone() / PUSH_LENGTH

            act_seq_i = torch.stack([xs_i, ys_i, thetas_i, lengths_i], dim=-1)
            act_seq_i = clip_actions(act_seq_i, action_lower_lim, action_upper_lim)
            act_seqs[1:, i] = act_seq_i[1:].clone()

    return act_seqs  # (n_sample, n_look_ahead, action_dim)

def clip_actions(action, action_lower_lim, action_upper_lim):
    action_new = action.clone()
    action_new[..., 2] = angle_normalize(action[..., 2])
    action_new.data.clamp_(action_lower_lim, action_upper_lim)
    return action_new

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

# ============ action optimization ============
def optimize_action_mppi(act_seqs, reward_seqs, reward_weight=100.0, action_lower_lim=None, action_upper_lim=None):
    weight_seqs = F.softmax(reward_seqs * reward_weight, dim=0).unsqueeze(-1)

    assert act_seqs.shape[-1] == 4  # (x, y, theta, length)
    xs = act_seqs[:, :, 0]
    ys = act_seqs[:, :, 1]
    thetas = act_seqs[:, :, 2]
    lengths = act_seqs[:, :, 3]
    x_ends = xs - lengths * PUSH_LENGTH * torch.cos(thetas)
    y_ends = ys - lengths * PUSH_LENGTH * torch.sin(thetas)

    x = torch.sum(weight_seqs * xs, dim=0)  # (n_look_ahead,)
    y = torch.sum(weight_seqs * ys, dim=0)  # (n_look_ahead,)
    x_end = torch.sum(weight_seqs * x_ends, dim=0)  # (n_look_ahead,)
    y_end = torch.sum(weight_seqs * y_ends, dim=0)  # (n_look_ahead,)

    theta = torch.atan2(y - y_end, x - x_end)  # (n_look_ahead,)
    length = torch.norm(torch.stack([x_end - x, y_end - y], dim=-1), dim=-1) / PUSH_LENGTH  # (n_look_ahead,)

    act_seq = torch.stack([x, y, theta, length], dim=-1)  # (n_look_ahead, action_dim)
    act_seq = clip_actions(act_seq, action_lower_lim, action_upper_lim)
    return act_seq

def point_in_box_2d(points, box):
    # points: (..., 2)
    # box: (2, 2)
    min_x, max_x = box[0]
    min_y, max_y = box[1]
    x_mask = (points[..., 0] >= min_x) & (points[..., 0] <= max_x)
    y_mask = (points[..., 1] >= min_y) & (points[..., 1] <= max_y)
    mask = x_mask & y_mask
    return mask

def running_cost(state, action, state_cur, material, target_specification=None, target_box=None, weights=None):
    # chamfer distance
    # state: (bsz, n_look_forward, max_nobj, 3)
    # action: (bsz, n_look_forward, action_dim)
    # target_specification: (target_indices (n,), target_state (n, 3))
    # state_cur: (max_nobj, 3)
    # weights: (n_look_forward,)
    bsz = state.shape[0]
    n_look_forward = state.shape[1]

    state_flat = state.reshape(bsz * n_look_forward, state.shape[2], state.shape[3])

    if target_specification is not None:
        target_specification = list(target_specification)
        target_specification[0] = torch.tensor(target_specification[0], device=state.device, dtype=torch.long)
        target_specification[1] = torch.tensor(target_specification[1], device=state.device, dtype=torch.float32)
        target_specification[1] = particle_real_to_sim(target_specification[1])

        # chamfer_distance = chamfer(state_flat, target_state).reshape(bsz, n_look_forward)
        # chamfer_distance = end_point_distance(state_flat, target_state).reshape(bsz, n_look_forward)
        # emd_distance = em_distance(state_flat, target_state).reshape(bsz, n_look_forward)
        # chamfer_distance = chamfer_distance + emd_distance
        distance = specification_square_distance(state_flat, target_specification).reshape(bsz, n_look_forward)
        if weights is None:
            distance = distance[:, -1]
        else:
            distance = torch.sum(weights * distance, dim=1)

    distance_weight = 2. / (distance.max().item() + 1e-6)

    if material == 'rope':
        x_start = action[:, :, 0]
        z_start = action[:, :, 1]
        action_point_2d = torch.stack([x_start, z_start], dim=-1)  # (bsz, n_look_forward, 2)
        state_2d = torch.cat([state_cur[:, [0, 2]][None, None].repeat(bsz, 1, 1, 1),
                              state[:, :-1, :, [0, 2]]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
        # state_2d = state_cur[:, [0, 2]][None, None]
        action_state_distance = torch.norm(action_point_2d[:, :, None] - state_2d, dim=-1).min(dim=-1).values  # (bsz, n_look_forward)
        pusher_size = 0.02 * SIM_REAL_RATIO
        action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
        collision_penalty = torch.exp(-action_state_distance * 100.)  # (bsz, n_look_forward) 
    elif material == 'cube':
        x_start = action[:, :, 0]  # (bsz, n_look_forward)
        z_start = action[:, :, 1]  # (bsz, n_look_forward)
        theta = action[:, :, 2]
        pusher_radius = 0.05 * SIM_REAL_RATIO
        delta_x = pusher_radius * torch.sin(theta)
        delta_z = -pusher_radius * torch.cos(theta)
        action_point_2d = torch.stack([
            x_start - delta_x, z_start - delta_z, 
            x_start - 0.75 * delta_x, z_start - 0.75 * delta_z,
            x_start - 0.5 * delta_x, z_start - 0.5 * delta_z,
            x_start - 0.25 * delta_x, z_start - 0.25 * delta_z,
            x_start, z_start,
            x_start + 0.25 * delta_x, z_start + 0.25 * delta_z,
            x_start + 0.5 * delta_x, z_start + 0.5 * delta_z,
            x_start + 0.75 * delta_x, z_start + 0.75 * delta_z,
            x_start + delta_x, z_start + delta_z], dim=-1)  # (bsz, n_look_forward, 9)
        state_2d = torch.cat([state_cur[:, [0, 2]][None, None].repeat(bsz, 1, 1, 1),
                              state[:, :-1, :, [0, 2]]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
        # state_2d = state_cur[:, [0, 2]][None, None]
        action_point_2d = action_point_2d.reshape(bsz, n_look_forward, 9, 2)  # (bsz, n_look_forward, 9, 2)
        action_state_distance = torch.norm(action_point_2d[:, :, :, None] - state_2d[:, :, None], dim=-1)  # (bsz, n_look_forward, 5, max_nobj)
        action_state_distance = action_state_distance.min(dim=-1).values.min(dim=-1).values  # (bsz, n_look_forward)
        pusher_size = 0.04 * SIM_REAL_RATIO # distance to centers
        action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
        collision_penalty = torch.exp(-action_state_distance * 1000.)  # (bsz, n_look_forward) 
    elif material == 'granular':
        x_start = action[:, :, 0]  # (bsz, n_look_forward)
        z_start = action[:, :, 1]  # (bsz, n_look_forward)
        theta = action[:, :, 2]
        pusher_radius = 0.05 * SIM_REAL_RATIO
        delta_x = pusher_radius * torch.sin(theta)
        delta_z = -pusher_radius * torch.cos(theta)
        action_point_2d = torch.stack([
            x_start - delta_x, z_start - delta_z, 
            x_start - 0.75 * delta_x, z_start - 0.75 * delta_z,
            x_start - 0.5 * delta_x, z_start - 0.5 * delta_z,
            x_start - 0.25 * delta_x, z_start - 0.25 * delta_z,
            x_start, z_start,
            x_start + 0.25 * delta_x, z_start + 0.25 * delta_z,
            x_start + 0.5 * delta_x, z_start + 0.5 * delta_z,
            x_start + 0.75 * delta_x, z_start + 0.75 * delta_z,
            x_start + delta_x, z_start + delta_z], dim=-1)  # (bsz, n_look_forward, 9)
        state_2d = torch.cat([state_cur[:, [0, 2]][None, None].repeat(bsz, 1, 1, 1),
                              state[:, :-1, :, [0, 2]]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
        # state_2d = state_cur[:, [0, 2]][None, None]
        action_point_2d = action_point_2d.reshape(bsz, n_look_forward, 9, 2)  # (bsz, n_look_forward, 9, 2)
        action_state_distance = torch.norm(action_point_2d[:, :, :, None] - state_2d[:, :, None], dim=-1)  # (bsz, n_look_forward, 5, max_nobj)
        action_state_distance = action_state_distance.min(dim=-1).values.min(dim=-1).values  # (bsz, n_look_forward)
        pusher_size = 0.04 * SIM_REAL_RATIO # distance to centers
        action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
        collision_penalty = torch.exp(-action_state_distance * 1000.)  # (bsz, n_look_forward) 
    else:
        raise NotImplementedError(f"material {material} not implemented")

    # bbox in simulator
    bbox = np.array([[0.3, 0.6], [-0.3, 0.3]]) * SIM_REAL_RATIO
    xmax = state.max(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    xmin = state.min(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    zmax = state.max(dim=2).values[:, :, 2]  # (bsz, n_look_forward)
    zmin = state.min(dim=2).values[:, :, 2]  # (bsz, n_look_forward)

    box_penalty = torch.stack([
        torch.maximum(xmin - bbox[0, 0], torch.zeros_like(xmin)),
        torch.maximum(bbox[0, 1] - xmax, torch.zeros_like(xmax)),
        torch.maximum(zmin - bbox[1, 0], torch.zeros_like(zmin)),
        torch.maximum(bbox[1, 1] - zmax, torch.zeros_like(zmax)),
    ], dim=-1)  # (bsz, n_look_forward, 4)
    box_penalty = torch.exp(-box_penalty * 1000.).max(dim=-1).values  # (bsz, n_look_forward)

    reward = -distance_weight * distance - 5. * collision_penalty.mean(dim=1) - 5. * box_penalty.mean(dim=1)  # (bsz,)

    print(f'min distance {distance.min().item()}, max reward {reward.max().item()}')
    out = {
        "reward_seqs": reward,
    }
    return out

# ============ dynamics ============
@torch.no_grad()
def dynamics(state, perturbed_action, model, material, device):
    time0 = time.time()
    max_n = 1
    max_nR = 2000
    n_his = 4

    bsz = perturbed_action.shape[0]
    n_look_forward = perturbed_action.shape[1]

    decoded_action, action_repeat = decode_action(perturbed_action, push_length=PUSH_LENGTH)

    obj_kp = state[None, None].repeat(bsz, n_his, 1, 1)
    obj_kp_num = obj_kp.shape[2]
    max_nobj = obj_kp_num

    pred_state_seq = torch.zeros((bsz, n_look_forward, max_nobj, 3), device=device)
    # pred_state_seq_all = torch.zeros((bsz, n_look_forward, action_repeat.max().item() + 1, max_nobj, 3), device=device)

    for li in range(n_look_forward):
        print(f"look forward iter {li}")
        
        if li > 0:
            obj_kp = pred_state_seq[:, li-1:li].detach().clone().repeat(1, n_his, 1, 1)

        # y = (obj_kp[:, -1, :, 1]).min(dim=1).values  # (bsz,)
        y = (obj_kp[:, -1, :, 1]).mean(dim=1)  # (bsz,)

        if material == 'rope':
            eef_kp_num = 1
            max_neef = eef_kp_num
            adj_thresh = 0.06 * SIM_REAL_RATIO
            eef_kp = torch.zeros((bsz, 1, 3))
            eef_kp[:, 0, 0] = decoded_action[:, li, 0]
            eef_kp[:, 0, 1] = y
            eef_kp[:, 0, 2] = decoded_action[:, li, 1]
            eef_kp_delta = torch.zeros((bsz, 1, 3))
            eef_kp_delta[:, 0, 0] = decoded_action[:, li, 2] - decoded_action[:, li, 0]
            eef_kp_delta[:, 0, 1] = 0
            eef_kp_delta[:, 0, 2] = decoded_action[:, li, 3] - decoded_action[:, li, 1]
        elif material == 'cube':
            eef_kp_num = 5
            max_neef = eef_kp_num
            adj_thresh = 0.04 * SIM_REAL_RATIO
            eef_kp = torch.zeros((bsz, 5, 3))
            eef_kp[:, :, 1] = y[:, None]
            eef_kp_delta = torch.zeros((bsz, 5, 3))
            eef_kp_delta[:, :, 0] = (decoded_action[:, li, 2] - decoded_action[:, li, 0]).unsqueeze(1)
            eef_kp_delta[:, :, 1] = 0
            eef_kp_delta[:, :, 2] = (decoded_action[:, li, 3] - decoded_action[:, li, 1]).unsqueeze(1)

            x_start = decoded_action[:, li, 0]
            z_start = decoded_action[:, li, 1]
            theta = perturbed_action[:, li, 2]

            eef_kp[:, 0, 0] = x_start
            eef_kp[:, 1, 0] = x_start + 0.05 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 2, 0] = x_start + 0.025 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 3, 0] = x_start - 0.025 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 4, 0] = x_start - 0.05 * SIM_REAL_RATIO * torch.sin(theta)

            eef_kp[:, 0, 2] = z_start
            eef_kp[:, 1, 2] = z_start - 0.05 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 2, 2] = z_start - 0.025 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 3, 2] = z_start + 0.025 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 4, 2] = z_start + 0.05 * SIM_REAL_RATIO * torch.cos(theta)
        elif material == 'granular':
            eef_kp_num = 5
            max_neef = eef_kp_num
            adj_thresh = 0.06 * SIM_REAL_RATIO
            eef_kp = torch.zeros((bsz, 5, 3))
            eef_kp[:, :, 1] = y[:, None]
            eef_kp_delta = torch.zeros((bsz, 5, 3))
            eef_kp_delta[:, :, 0] = (decoded_action[:, li, 2] - decoded_action[:, li, 0]).unsqueeze(1)
            eef_kp_delta[:, :, 1] = 0
            eef_kp_delta[:, :, 2] = (decoded_action[:, li, 3] - decoded_action[:, li, 1]).unsqueeze(1)

            x_start = decoded_action[:, li, 0]
            z_start = decoded_action[:, li, 1]
            theta = perturbed_action[:, li, 2]

            eef_kp[:, 0, 0] = x_start
            eef_kp[:, 1, 0] = x_start + 0.05 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 2, 0] = x_start + 0.025 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 3, 0] = x_start - 0.025 * SIM_REAL_RATIO * torch.sin(theta)
            eef_kp[:, 4, 0] = x_start - 0.05 * SIM_REAL_RATIO * torch.sin(theta)

            eef_kp[:, 0, 2] = z_start
            eef_kp[:, 1, 2] = z_start - 0.05 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 2, 2] = z_start - 0.025 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 3, 2] = z_start + 0.025 * SIM_REAL_RATIO * torch.cos(theta)
            eef_kp[:, 4, 2] = z_start + 0.05 * SIM_REAL_RATIO * torch.cos(theta)
        else:
            raise NotImplementedError(f"material {material} not implemented")

        states = torch.zeros((bsz, n_his, max_nobj + max_neef, 3), device=device)
        states[:, :, :obj_kp_num] = obj_kp
        states[:, :, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, None]

        states_delta = torch.zeros((bsz, max_nobj + max_neef, 3), device=device)
        states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp_delta

        attr_dim = 2
        attrs = torch.zeros((bsz, max_nobj + max_neef, attr_dim), dtype=torch.float32, device=device)
        attrs[:, :obj_kp_num, 0] = 1.
        attrs[:, max_nobj : max_nobj + eef_kp_num, 1] = 1.

        p_rigid = torch.zeros((bsz, max_n), dtype=torch.float32, device=device)

        p_instance = torch.zeros((bsz, max_nobj, max_n), dtype=torch.float32, device=device)
        instance_num = 1
        instance_kp_nums = [obj_kp_num]
        for i in range(bsz):
            ptcl_cnt = 0
            for j in range(instance_num):
                p_instance[i, ptcl_cnt:ptcl_cnt + instance_kp_nums[j], j] = 1
                ptcl_cnt += instance_kp_nums[j]

        state_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        state_mask[:, max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:, :obj_kp_num] = True

        eef_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        eef_mask[:, max_nobj : max_nobj + eef_kp_num] = True

        obj_mask = torch.zeros((bsz, max_nobj,), dtype=bool, device=device)
        obj_mask[:, :obj_kp_num] = True

        pushing_direction = decoded_action[:, li, 2:4] - decoded_action[:, li, :2]
        pushing_direction = torch.cat([pushing_direction[:, 0:1], torch.zeros_like(pushing_direction[:, 0:1]), pushing_direction[:, 1:2]], dim=-1)

        Rr, Rs = construct_edges_from_states(states[:, -1], adj_thresh, 
                    mask=state_mask, tool_mask=eef_mask, no_self_edge=True, pushing_direction=pushing_direction)  # pushing_direction=None, 
        Rr = pad_torch(Rr, max_nR, dim=1)
        Rs = pad_torch(Rs, max_nR, dim=1)

        graph = {
            # input information
            "state": states,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_rigid": p_rigid,  # (n_instance,)
            "p_instance": p_instance,  # (N, n_instance)
            "obj_mask": obj_mask,  # (N,)
            "state_mask": state_mask,  # (N+M,)
            "eef_mask": eef_mask,  # (N+M,)

            "Rr": Rr,  # (bsz, max_nR, N)
            "Rs": Rs,  # (bsz, max_nR, N)
        }

        # pred_state_seq_all[:, li, 0] = state[None].detach().clone()

        # rollout
        for ai in range(1, 1 + action_repeat[:, li].max().item()):
            # print(f"rollout iter {i}")
            graph = truncate_graph(graph)
            pred_state, pred_motion = model(**graph)

            repeat_mask = (action_repeat[:, li] == ai)
            pred_state_seq[repeat_mask, li] = pred_state[repeat_mask, :, :].clone()
            # pred_state_seq_all[:, li, ai] = pred_state.detach().clone()

            # y_cur = pred_state[:, :, 1].min(dim=1).values
            y_cur = pred_state[:, :, 1].mean(dim=1)
            eef_kp_cur = graph['state'][:, -1, max_nobj : max_nobj + eef_kp_num] + graph['action'][:, max_nobj : max_nobj + eef_kp_num]

            if material == 'rope':
                eef_kp_cur[:, 0, 1] = y_cur
            elif material == 'cube':
                eef_kp_cur[:, :, 1] = y_cur[:, None]
            elif material == 'granular':
                eef_kp_cur[:, :, 1] = y_cur[:, None]
            else:
                raise NotImplementedError(f"material {material} not implemented")

            states_cur = torch.cat([pred_state, eef_kp_cur], dim=1)
            Rr, Rs = construct_edges_from_states(states_cur, adj_thresh, 
                        mask=graph['state_mask'], tool_mask=graph['eef_mask'], no_self_edge=True, pushing_direction=pushing_direction)  # pushing_direction=None, 
            Rr = pad_torch(Rr, max_nR, dim=1)
            Rs = pad_torch(Rs, max_nR, dim=1)

            # action encoded as state_delta (only stored in eef keypoints)
            # states_delta = torch.zeros((bsz, max_nobj + max_neef, states.shape[-1]), dtype=torch.float32, device=device)
            # states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, 1] - eef_kp[:, 0]

            state_history = torch.cat([graph['state'][:, 1:], states_cur[:, None]], dim=1)

            new_graph = {
                "state": state_history,  # (bsz, n_his, N+M, state_dim)
                "action": graph["action"],  # (bsz, N+M, state_dim)
                
                "Rr": Rr,  # (bsz, n_rel, N+M)
                "Rs": Rs,  # (bsz, n_rel, N+M)
                
                "attrs": graph["attrs"],  # (bsz, N+M, attr_dim)
                "p_rigid": graph["p_rigid"],  # (bsz, n_instance)
                "p_instance": graph["p_instance"],  # (bsz, N, n_instance)
                "obj_mask": graph["obj_mask"],  # (bsz, N)
                "eef_mask": graph["eef_mask"],  # (bsz, N+M)
                "state_mask": graph["state_mask"],  # (bsz, N+M)
            }
            
            graph = new_graph

    out = {
        "state_seqs": pred_state_seq,  # (bsz, n_look_forward, max_nobj, 3)
        "action_seqs": decoded_action,  # (bsz, n_look_forward, action_dim)
        # "action_repeat": action_repeat,  # (bsz, n_look_forward)
        # "state_seqs_all": pred_state_seq_all,  # (bsz, n_look_forward, max_repeat+1, max_nobj, 3)
    }
    time1 = time.time()
    print(f"dynamics time {time1 - time0}")
    return out

def dynamics_chunk(state, perturbed_action, model, material, device, n_sample_chunk):
    # state: (max_nobj, 3)
    # perturbed_action: (n_sample, n_look_forward, action_dim)
    n_samples = perturbed_action.shape[0]
    n_chunks = np.ceil(n_samples / n_sample_chunk).astype(int)
    for ci in range(n_chunks):
        start = ci * n_sample_chunk
        end = min((ci + 1) * n_sample_chunk, n_samples)
        perturbed_action_chunk = perturbed_action[start:end]
        out = dynamics(state, perturbed_action_chunk, model, material, device)
        if ci == 0:
            res = out
        else:
            # "state_seqs" (bsz, n_look_forward, max_nobj, 3)
            # "action_seqs" (bsz, n_look_forward, action_dim)
            for k, v in out.items():
                res[k] = torch.cat([res[k], v], dim=0)
    return res

# ============ visualization ============
def view_specification(img, state, specification, intr, extr):
    # add specification to img
    def project(points, intr, extr):
        # extr: (4, 4)
        # intr: (3, 3)
        # points: (n_points, 3)

        # transform points back to real coordinates
        points = particle_sim_to_real(points)

        # project
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = points @ np.linalg.inv(extr).T  # (n_points, 4)
        points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
        points = points @ intr.T
        points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
        return points

    img = img.copy()
    target_indices, target_points = specification
    target_image_points = project(particle_real_to_sim(target_points), intr, extr)
    state_image = project(state.detach().cpu().numpy(), intr, extr)

    for index, target_image_point in zip(target_indices, target_image_points):
        x, y = state_image[index]
        x_target, y_target = target_image_point

        # draw a little cross at the target point
        cv2.line(img, (int(x_target - 3), int(y_target)), (int(x_target + 3), int(y_target)), (0, 255, 0), 2)
        cv2.line(img, (int(x_target), int(y_target - 3)), (int(x_target), int(y_target + 3)), (0, 255, 0), 2)
        # draw an arrow from the current point to the target point
        cv2.arrowedLine(img, (int(x), int(y)), (int(x_target), int(y_target)), (0, 255, 0), 2)

    return img

def visualize_img(state_init, res, rgb_vis, material, intr, extr, save_dir=None, postfix=None):
    # add current state, action and prediction to rgb_vis
    # state_init: (n_points, 3)
    # state: (n_look_forward, n_points, 3)
    # rgb_vis: np.ndarray (H, W, 3)

    def project(points, intr, extr):
        # extr: (4, 4)
        # intr: (3, 3)
        # points: (n_points, 3)

        # transform points back to real coordinates
        points = particle_sim_to_real(points)

        # project
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = points @ np.linalg.inv(extr).T  # (n_points, 4)
        points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
        points = points @ intr.T
        points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
        return points

    # best result
    action_best = res['act_seq']  # (n_look_forward, action_dim)
    state_best = res['best_model_output']['state_seqs'][0]  # (n_look_forward, max_nobj, 3)

    # construct relations
    if material == 'cube':
        adj_thresh = 0.04 * SIM_REAL_RATIO  # TODO 1d
    elif material == 'granular':
        adj_thresh = 0.06 * SIM_REAL_RATIO
    elif material == 'rope':
        adj_thresh = 0.06 * SIM_REAL_RATIO
    elif material == 'cloth':
        adj_thresh = 0.075 * SIM_REAL_RATIO
    else:
        raise NotImplementedError(f"material {material} not implemented")
    
    # plot
    action, repeat = decode_action(action_best.unsqueeze(0), push_length=PUSH_LENGTH)  # (1, n_look_forward, action_dim)
    action = action[0]  # (n_look_forward, action_dim)
    repeat = repeat[0, 0].item()

    state_init_vis = state_init.detach().cpu().numpy()  # (n_points, 3)
    state_vis = state_best[0].detach().cpu().numpy()  # (n_points, 3)
    action_vis = action[0].detach().cpu().numpy()  # (action_dim,)

    # pushing_direction = action_vis[2:4] - action_vis[:2]
    # pushing_direction = np.concatenate([pushing_direction[:1], np.zeros(1), pushing_direction[1:2]], axis=-1)
    # pushing_direction = torch.from_numpy(pushing_direction)

    Rr, Rs = construct_edges_from_states(torch.from_numpy(state_init_vis)[None], adj_thresh, 
                mask=torch.ones(state_init_vis.shape[0], dtype=bool)[None],
                tool_mask=torch.zeros(state_init_vis.shape[0], dtype=bool)[None], no_self_edge=True,
                pushing_direction=None)
                # pushing_direction=pushing_direction)
    Rr = Rr[0].numpy()  # (n_rel, n_points)
    Rs = Rs[0].numpy()  # (n_rel, n_points)

    # Rr_best = Rr.copy()
    # Rs_best = Rs.copy()

    Rr_best, Rs_best = construct_edges_from_states(torch.from_numpy(state_vis)[None], adj_thresh,
                mask=torch.ones(state_vis.shape[0], dtype=bool)[None],
                tool_mask=torch.zeros(state_vis.shape[0], dtype=bool)[None], no_self_edge=True,
                pushing_direction=None)
                # pushing_direction=pushing_direction)
    Rr_best = Rr_best[0].numpy()  # (n_rel, n_points)
    Rs_best = Rs_best[0].numpy()  # (n_rel, n_points)

    # plot state_init_vis, Rr, Rs, action_vis, state_vis, target_state_vis on rgb_vis

    # preparation
    state_init_proj = project(state_init_vis, intr, extr)
    state_proj = project(state_vis, intr, extr)

    # visualize
    rgb_orig = rgb_vis.copy()
    # cv2.imwrite(os.path.join(save_dir, f'rgb_original_{postfix}.png'), rgb_orig)

    color_start = (202, 63, 41)
    color_action = (27, 74, 242)
    color_pred = (237, 158, 49)

    # starting state
    point_size = 3
    for k in range(state_init_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_init_proj[k, 0]), int(state_init_proj[k, 1])), point_size, 
            color_start, -1)

    # starting edges
    edge_size = 1
    for k in range(Rr.shape[0]):
        if Rr[k].sum() == 0: continue
        receiver = Rr[k].argmax()
        sender = Rs[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_init_proj[receiver, 0]), int(state_init_proj[receiver, 1])), 
            (int(state_init_proj[sender, 0]), int(state_init_proj[sender, 1])), 
            color_start, edge_size)
    
    # action arrow
    x_start = action_vis[0]
    z_start = action_vis[1]
    x_end = action_vis[2]
    z_end = action_vis[3]
    x_delta = x_end - x_start
    z_delta = z_end - z_start
    y = state_init[:, 1].mean().item()
    if material == 'cloth':
        y += 0.01 * SIM_REAL_RATIO
    arrow_size = 3
    tip_length = 0.5
    for i in range(repeat):
        action_start_point = np.array([x_start + i * x_delta, y, z_start + i * z_delta])
        action_end_point = np.array([x_end + i * x_delta, y, z_end + i * z_delta])
        action_start_point_proj = project(action_start_point[None], intr, extr)[0]
        action_end_point_proj = project(action_end_point[None], intr, extr)[0]
        cv2.arrowedLine(rgb_vis,
            (int(action_start_point_proj[0]), int(action_start_point_proj[1])),
            (int(action_end_point_proj[0]), int(action_end_point_proj[1])),
            color_action, arrow_size, tipLength=tip_length)

    rgb_overlay = rgb_vis.copy()

    # predicted state
    for k in range(state_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_proj[k, 0]), int(state_proj[k, 1])), point_size, 
            color_pred, -1)
    
    # predicted edges
    for k in range(Rr_best.shape[0]):
        if Rr_best[k].sum() == 0: continue
        receiver = Rr_best[k].argmax()
        sender = Rs_best[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_proj[receiver, 0]), int(state_proj[receiver, 1])), 
            (int(state_proj[sender, 0]), int(state_proj[sender, 1])), 
            color_pred, edge_size)
    
    rgb_vis = cv2.addWeighted(rgb_overlay, 0.5, rgb_vis, 0.5, 0)
    
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'rgb_vis_{postfix}.png'), rgb_vis)
        cv2.imwrite(os.path.join(save_dir, f'rgb_orig_{postfix}.png'), rgb_orig)
    
    # optionally plot obj_pcd on rgb_vis
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obj_pcd)
    # visualize_o3d([pcd])

# for debug
def visualize_push(state_init, res, model, rgb_vis, material, intr, extr, device, save_dir=None, postfix=None):
    # state_init: (n_points, 3)
    # rgb_vis: np.ndarray (H, W, 3)
    max_n = 1
    max_nR = 2000
    n_his = 4

    # visualize
    color = (202, 63, 41)
    point_size = 3
    edge_size = 1

    def project(points, intr, extr):
        # extr: (4, 4)
        # intr: (3, 3)
        # points: (n_points, 3)

        # transform points back to real coordinates
        points = particle_sim_to_real(points)

        # project
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = points @ np.linalg.inv(extr).T  # (n_points, 4)
        points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
        points = points @ intr.T
        points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
        return points

    # action
    action_best = res['act_seq']  # (n_look_forward, action_dim)
    action_best = action_best[None]
    n_look_forward = 1 # only support n_look_forward=1
    decoded_action, action_repeat = decode_action(action_best, push_length=PUSH_LENGTH)

    obj_kp = state_init[None, None].repeat(1, n_his, 1, 1)
    obj_kp_num = obj_kp.shape[2]
    max_nobj = obj_kp_num

    y = (obj_kp[:, -1, :, 1]).mean(dim=1)

    if material == 'rope':
        eef_kp_num = 1
        max_neef = eef_kp_num
        adj_thresh = 0.06 * SIM_REAL_RATIO
        eef_kp = torch.zeros((1, 1, 3))
        eef_kp[:, 0, 0] = decoded_action[:, 0, 0]  # TODO batch
        eef_kp[:, 0, 1] = y
        eef_kp[:, 0, 2] = decoded_action[:, 0, 1]
        eef_kp_delta = torch.zeros((1, 1, 3))
        eef_kp_delta[:, 0, 0] = decoded_action[:, 0, 2] - decoded_action[:, 0, 0]
        eef_kp_delta[:, 0, 1] = 0
        eef_kp_delta[:, 0, 2] = decoded_action[:, 0, 3] - decoded_action[:, 0, 1]
    elif material == 'granular':
        eef_kp_num = 5
        max_neef = eef_kp_num
        adj_thresh = 0.06 * SIM_REAL_RATIO
        eef_kp = torch.zeros((1, 5, 3))
        eef_kp[:, :, 1] = y[:, None]
        # eef_kp[:, :, 0] = decoded_action[:, li, 0]  # TODO batch
        # eef_kp[:, :, 2] = decoded_action[:, li, 1]
        eef_kp_delta = torch.zeros((1, 5, 3))
        eef_kp_delta[:, :, 0] = (decoded_action[:, 0, 2] - decoded_action[:, 0, 0]).unsqueeze(1)
        eef_kp_delta[:, :, 1] = 0
        eef_kp_delta[:, :, 2] = (decoded_action[:, 0, 3] - decoded_action[:, 0, 1]).unsqueeze(1)

        x_start = decoded_action[:, 0, 0]
        z_start = decoded_action[:, 0, 1]
        theta = action_best[:, 0, 2]

        eef_kp[:, 0, 0] = x_start
        eef_kp[:, 1, 0] = x_start + 0.05 * SIM_REAL_RATIO * torch.sin(theta)
        eef_kp[:, 2, 0] = x_start + 0.025 * SIM_REAL_RATIO * torch.sin(theta)
        eef_kp[:, 3, 0] = x_start - 0.025 * SIM_REAL_RATIO * torch.sin(theta)
        eef_kp[:, 4, 0] = x_start - 0.05 * SIM_REAL_RATIO * torch.sin(theta)

        eef_kp[:, 0, 2] = z_start
        eef_kp[:, 1, 2] = z_start - 0.05 * SIM_REAL_RATIO * torch.cos(theta)
        eef_kp[:, 2, 2] = z_start - 0.025 * SIM_REAL_RATIO * torch.cos(theta)
        eef_kp[:, 3, 2] = z_start + 0.025 * SIM_REAL_RATIO * torch.cos(theta)
        eef_kp[:, 4, 2] = z_start + 0.05 * SIM_REAL_RATIO * torch.cos(theta)

    states = torch.zeros((1, n_his, max_nobj + max_neef, 3), device=device)
    states[:, :, :obj_kp_num] = obj_kp
    states[:, :, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, None]

    states_delta = torch.zeros((1, max_nobj + max_neef, 3), device=device)
    states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp_delta

    attr_dim = 2
    attrs = torch.zeros((1, max_nobj + max_neef, attr_dim), dtype=torch.float32, device=device)
    attrs[:, :obj_kp_num, 0] = 1.
    attrs[:, max_nobj : max_nobj + eef_kp_num, 1] = 1.

    p_rigid = torch.zeros((1, max_n), dtype=torch.float32, device=device)

    p_instance = torch.zeros((1, max_nobj, max_n), dtype=torch.float32, device=device)
    instance_num = 1
    instance_kp_nums = [obj_kp_num]
    for i in range(1):
        ptcl_cnt = 0
        for j in range(instance_num):
            p_instance[i, ptcl_cnt:ptcl_cnt + instance_kp_nums[j], j] = 1
            ptcl_cnt += instance_kp_nums[j]

    state_mask = torch.zeros((1, max_nobj + max_neef), dtype=bool, device=device)
    state_mask[:, max_nobj : max_nobj + eef_kp_num] = True
    state_mask[:, :obj_kp_num] = True

    eef_mask = torch.zeros((1, max_nobj + max_neef), dtype=bool, device=device)
    eef_mask[:, max_nobj : max_nobj + eef_kp_num] = True

    obj_mask = torch.zeros((1, max_nobj,), dtype=bool, device=device)
    obj_mask[:, :obj_kp_num] = True

    pushing_direction = decoded_action[:, 0, 2:4] - decoded_action[:, 0, :2]
    pushing_direction = torch.cat([pushing_direction[:, 0:1], torch.zeros_like(pushing_direction[:, 0:1]), pushing_direction[:, 1:2]], dim=-1)

    Rr, Rs = construct_edges_from_states(states[:, -1], adj_thresh, 
                mask=state_mask, tool_mask=eef_mask, no_self_edge=True, pushing_direction=pushing_direction)  # pushing_direction=None, 
    Rr = pad_torch(Rr, max_nR, dim=1)
    Rs = pad_torch(Rs, max_nR, dim=1)

    # draw the current state
    states_vis = states[0, -1].detach().cpu().numpy()
    states_proj = project(states_vis, intr, extr)
    rgb_tem = rgb_vis.copy()
    for k in range(states_proj.shape[0]):
        cv2.circle(rgb_tem, (int(states_proj[k, 0]), int(states_proj[k, 1])), point_size, color, -1)
    for k in range(Rr[0].shape[0]):
        if Rr[0][k].sum() == 0: continue
        receiver = Rr[0][k].argmax().item()
        sender = Rs[0][k].argmax().item()
        cv2.line(rgb_tem, 
            (int(states_proj[receiver, 0]), int(states_proj[receiver, 1])), 
            (int(states_proj[sender, 0]), int(states_proj[sender, 1])), 
            color, edge_size)
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'push_vis_{postfix}_0.png'), rgb_tem)

    graph = {
        # input information
        "state": states,  # (n_his, N+M, state_dim)
        "action": states_delta,  # (N+M, state_dim)

        # attr information
        "attrs": attrs,  # (N+M, attr_dim)
        "p_rigid": p_rigid,  # (n_instance,)
        "p_instance": p_instance,  # (N, n_instance)
        "obj_mask": obj_mask,  # (N,)
        "state_mask": state_mask,  # (N+M,)
        "eef_mask": eef_mask,  # (N+M,)

        "Rr": Rr,  # (bsz, max_nR, N)
        "Rs": Rs,  # (bsz, max_nR, N)
    }

    # rollout and visualize
    for ai in range(1, 1 + action_repeat[:, 0].max().item()):
        graph = truncate_graph(graph)
        pred_state, pred_motion = model(**graph)

        y_cur = pred_state[:, :, 1].mean(dim=1)
        eef_kp_cur = graph['state'][:, -1, max_nobj : max_nobj + eef_kp_num] + graph['action'][:, max_nobj : max_nobj + eef_kp_num]

        if material == 'rope':
            eef_kp_cur[:, 0, 1] = y_cur

        states_cur = torch.cat([pred_state, eef_kp_cur], dim=1)
        Rr, Rs = construct_edges_from_states(states_cur, adj_thresh,
                    mask=graph['state_mask'], tool_mask=graph['eef_mask'], no_self_edge=True, pushing_direction=pushing_direction)
        Rr = pad_torch(Rr, max_nR, dim=1)
        Rs = pad_torch(Rs, max_nR, dim=1)

        # visualize
        states_cur_vis = states_cur[0].detach().cpu().numpy()
        states_cur_proj = project(states_cur_vis, intr, extr)
        rgb_tem = rgb_vis.copy()
        for k in range(states_cur_proj.shape[0]):
            cv2.circle(rgb_tem, (int(states_cur_proj[k, 0]), int(states_cur_proj[k, 1])), point_size, color, -1)
        for k in range(Rr[0].shape[0]):
            if Rr[0][k].sum() == 0: continue
            receiver = Rr[0][k].argmax().item()
            sender = Rs[0][k].argmax().item()
            cv2.line(rgb_tem, 
                (int(states_cur_proj[receiver, 0]), int(states_cur_proj[receiver, 1])), 
                (int(states_cur_proj[sender, 0]), int(states_cur_proj[sender, 1])), 
                color, edge_size)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, f'push_vis_{postfix}_{ai}.png'), rgb_tem)

        state_history = torch.cat([graph['state'][:, 1:], states_cur[:, None]], dim=1)

        new_graph = {
            "state": state_history,  # (bsz, n_his, N+M, state_dim)
            "action": graph["action"],  # (bsz, N+M, state_dim)
            
            "Rr": Rr,  # (bsz, n_rel, N+M)
            "Rs": Rs,  # (bsz, n_rel, N+M)
            
            "attrs": graph["attrs"],  # (bsz, N+M, attr_dim)
            "p_rigid": graph["p_rigid"],  # (bsz, n_instance)
            "p_instance": graph["p_instance"],  # (bsz, N, n_instance)
            "obj_mask": graph["obj_mask"],  # (bsz, N)
            "eef_mask": graph["eef_mask"],  # (bsz, N+M)
            "state_mask": graph["state_mask"],  # (bsz, N+M)
        }

        graph = new_graph

# ============ state ============
def get_state_cur(object, env, device, top_down_cam=None, fps_radius=0.02, visualize=False):
    object_pcds_list = [env.get_points_by_name(object, camera_index=top_down_cam, debug=visualize)]
    object_pcds = [item for sublist in object_pcds_list for item in sublist]
    obj_kps_list = truncate_points(object_pcds, fps_radius)
    obj_kps = np.concatenate(obj_kps_list, axis=0)
    state_cur = particle_real_to_sim(obj_kps)
    state_cur = torch.tensor(state_cur, dtype=torch.float32, device=device)

    rgb_vis = env.get_rgb_depth_pc()[0][0]

    return state_cur, obj_kps, rgb_vis

# ============ MOTION PLANNING ============
def closed_loop_plan(
        points,
        target_specification,
        object,
        material,
        env,
        top_down_cam,
        tracking_cam,
        save_dir,
        track_as_state=True,
        target_pcd=None,
    ):
    start_time_full = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bbox = env.get_bbox()[:2]
    bbox[:, 0] += 0.05
    bbox[:, 1] -= 0.05

    # transfor bbox into simulator
    if material == 'rope':
        bbox *= SIM_REAL_RATIO
        action_lower_lim = [
            bbox[0, 0], # x min
            -bbox[1, 1], # z min
            -math.pi,
            3,
        ]
        action_upper_lim = [
            bbox[0, 1], # x max
            -bbox[1, 0], # z max
            math.pi,
            15,
        ]
    elif material == 'cube':
        bbox *= SIM_REAL_RATIO
        action_lower_lim = [
            bbox[0, 0],
            -bbox[1, 1],
            -math.pi,
            3,
        ]
        action_upper_lim = [
            bbox[0, 1],
            -bbox[1, 0],
            math.pi,
            20,
        ]
    elif material == 'granular':
        bbox *= SIM_REAL_RATIO
        action_lower_lim = [
            bbox[0, 0],
            -bbox[1, 1],
            -math.pi,
            3,
        ]
        action_upper_lim = [
            bbox[0, 1],
            -bbox[1, 0],
            math.pi,
            25,
        ]
    else:
        raise NotImplementedError(f"material {material} not implemented")
    action_lower_lim = torch.tensor(action_lower_lim, dtype=torch.float32, device=device)
    action_upper_lim = torch.tensor(action_upper_lim, dtype=torch.float32, device=device)

    config, ckpt = load_config_and_ckpt(material)
    plan_config = config['plan']
    model_config = config['model']

    set_seed(plan_config['seed'])

    save_dir = f'{save_dir}/{material}-planning-{time.time()}'
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    model = DynamicsPredictor(model_config, device)
    end_time = time.time()
    print("Total time to initialize Dynamics model: {:.2f} seconds".format(end_time - start_time))

    model.to(device)

    model.eval()
    model.load_state_dict(ckpt)
    
    start_time = time.time()
    tracker = SpaTrackerWrapper()
    end_time = time.time()
    print("Total time to initialize Spatracker: {:.2f} seconds".format(end_time - start_time))

    running_cost_func = partial(running_cost, material=material, target_specification=target_specification)

    n_actions = plan_config['n_actions'] # total horizon size
    n_look_ahead = plan_config['n_look_ahead'] # sliding window size
    n_sample = plan_config['n_sample']
    n_sample_chunk = plan_config['n_sample_chunk']
    noise_level = plan_config['noise_level']
    reward_weight = plan_config['reward_weight']

    n_chunk = np.ceil(n_sample / n_sample_chunk).astype(int)

    planner_config = {
        'action_dim': len(action_lower_lim),
        # 'state_dim': max_nobj * 3 + max_neef * 3,
        'model_rollout_fn': partial(dynamics, model=model, device=device, material=material),
        # 'evaluate_traj_fn': partial(running_cost_straighten, material=material),
        'evaluate_traj_fn': running_cost_func,
        'sampling_action_seq_fn': partial(sample_action_seq, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim, 
                                        n_sample=min(n_sample, n_sample_chunk), device=device, noise_level=noise_level),
        'clip_action_seq_fn': partial(clip_actions, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim),
        'optimize_action_mppi_fn': partial(optimize_action_mppi, reward_weight=reward_weight, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim),
        'n_sample': min(n_sample, n_sample_chunk),
        'n_look_ahead': n_look_ahead,
        'n_update_iter': 1,
        'reward_weight': reward_weight,
        'action_lower_lim': action_lower_lim,
        'action_upper_lim': action_upper_lim,
        'planner_type': 'MPPI',
        'device': device,
        'verbose': False,
        'noise_level': noise_level,
        'rollout_best': True,
    }
    planner = Planner(planner_config)
    planner.total_chunks = n_chunk

    act_seq = torch.rand((planner_config['n_look_ahead'], action_upper_lim.shape[0]), device=device) * \
                (action_upper_lim - action_lower_lim) + action_lower_lim

    res_act_seq = torch.zeros((n_actions, action_upper_lim.shape[0]), device=device)
    end_time_full = time.time()
    print("Total time to initialize close loop planner config: {:.2f} seconds".format(end_time_full - start_time_full))
    for i in range(n_actions):
        time1 = time.time()

        # get state
        if i == 0:
            state_cur = particle_real_to_sim(points)
            state_cur = torch.tensor(state_cur, dtype=torch.float32, device=device)

        # get action
        res_all = []
        for ci in range(n_chunk):
            planner.chunk_id = ci
            res = planner.trajectory_optimization(state_cur, act_seq)
            for k, v in res.items():
                res[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
            res_all.append(res)
        res = planner.merge_res(res_all)

        # vis
        obs = env.get_rgb_depth_pc()[0][top_down_cam]
        intr = env.get_intrinsic(top_down_cam)
        extr = env.get_extrinsic(top_down_cam)
        vis_img = view_specification(obs, state_cur, target_specification, intr, extr)
        cv2.imwrite(os.path.join(save_dir, f'rgb_original_{i}_0.png'), obs)
        visualize_img(state_cur, res, vis_img, material, intr, extr,
                    save_dir=save_dir, postfix=f'{i}_0')
        # visualize_push(state_cur, res, model, obs, material, intr, extr, device, save_dir=save_dir, postfix=f'{i}_0')
        print('begin action')

        # step action
        video_recorder = VideoRecorder(
            index=tracking_cam,
            camera=env.realsense,
            capture_fps=15,
            record_fps=15,
            record_time=40,
            save_path=f'{save_dir}/tracking_{i}',
        )
        action = res['act_seq'][0].detach().cpu().numpy()
        step_and_record(env, action, video_recorder)
        # update action
        res_act_seq[i] = res['act_seq'][0].detach().clone()
        act_seq = torch.cat(
            [
                res['act_seq'][1:],
                torch.rand((1, action_upper_lim.shape[0]), device=device) * (action_upper_lim - action_lower_lim) + action_lower_lim
            ], 
            dim=0
        )
        n_look_ahead = min(n_actions - i, planner_config['n_look_ahead'])
        act_seq = act_seq[:n_look_ahead]  # sliding window
        planner.n_look_ahead = n_look_ahead

        # measurement
        if target_pcd is not None:
            print(target_pcd)
            # read from .pcd
            # target_pcd = o3d.io.read_point_cloud(target_pcd)
            # target_pcd = np.asarray(target_pcd.points)
            # np.unsqueeze
            # target_pcd = target_pcd[np.newaxis, ...]
            object_pcd = env.get_points_by_name(material, camera_index=top_down_cam, fuse=True)
            object_pcd = np.asarray(object_pcd)
            chamfer_dist = chamfer(target_pcd, object_pcd)[0]
            print(f"chamfer distance: {chamfer_dist}")
            # log
            with open(f'{save_dir}/chamfer_log.txt', 'a') as f:
                f.write(f"chamfer distance: {chamfer_dist}\n")

        # tracking
        queries = particle_sim_to_real(state_cur).detach().cpu().numpy()
        queries = env.world_to_viewport(queries, tracking_cam)
        video_dir = f'{save_dir}/tracking_{i}'
        frames = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        # video = []
        # for frame in frames:
        #     img = Image.open(frame)
        #     img = np.array(img)
        #     video.append(img)
        # video = np.stack(video, axis=0)
        # depths = []
        # depth_files = sorted(glob.glob(os.path.join(video_dir, '*.png')))
        # for depth_file in depth_files:
        #     depth = Image.open(depth_file)
        #     depth = np.array(depth) / 1000.
        #     depths.append(depth)
        # depths = np.stack(depths, axis=0)
        # video = video[:len(depths)] # sometimes the last depth frame is not saved
        #get every 10th video element in the video list
        # for i in range(len(depths)):
        #     video.append(video[i * 10])  # repeat every 10th frame
        # video = np.array(video)

        frames = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))[::5]
        video = []
        for frame in frames:
            img = Image.open(frame)
            video.append(np.array(img))
        video = np.stack(video, axis=0)

        # get every 10th depth frame
        depth_files = sorted(glob.glob(os.path.join(video_dir, '*.png')))[::5]
        depths = []
        for depth_file in depth_files:
            depth = Image.open(depth_file)
            depths.append(np.array(depth) / 1000.)
        depths = np.stack(depths, axis=0)

        pred_tracks, pred_visibility = tracker(video, depths, queries, debug=True, save_dir=video_dir) # pred_tracks: (T, N, 2)

        if pred_tracks.ndim == 4:
            pred_tracks = pred_tracks[:, 0]
        start_time = time.time()
        # update state and target
        depth = env.get_rgb_depth_pc()[2][tracking_cam]
        if track_as_state:
            state_cur = []
            for point in pred_tracks[-1]:
                x, y = point
                ground_point = env.ground_position(tracking_cam, depth, x, y)
                state_cur.append(ground_point)
            state_cur = np.array(state_cur)
            state_cur = particle_real_to_sim(state_cur)
            state_cur = torch.tensor(state_cur, dtype=torch.float32, device=device)
        else:
            state_cur, obj_kps, _ = get_state_cur(object, env, device, top_down_cam=top_down_cam)
            # change target_specification
            new_indices = []
            for index in target_specification[0]:
                x, y = pred_tracks[-1][index]
                ground_point = env.ground_position(tracking_cam, depth, x, y)
                # find the nearest point in obj_kps
                dists = np.linalg.norm(ground_point - obj_kps, axis=1)
                # mask to avoid duplicate
                dists[new_indices] = 1e6
                nearest_index = np.argmin(dists)
                new_indices.append(nearest_index)
            target_specification = (new_indices, target_specification[1])
            running_cost_func = partial(running_cost, material=material, target_specification=target_specification)
            planner.evaluate_traj = running_cost_func
        end_time = time.time()
        print("Total time to update state and target: {:.2f} seconds".format(end_time - start_time))
        # vis
        obs = env.get_rgb_depth_pc()[0][top_down_cam]
        vis_img = view_specification(obs, state_cur, target_specification, intr, extr)
        cv2.imwrite(os.path.join(save_dir, f'rgb_original_{i}_1.png'), obs)
        visualize_img(state_cur, res, vis_img, material, intr, extr,
                    save_dir=save_dir, postfix=f'{i}_1')

        time2 = time.time()
        print(f"step {i} time {time2 - time1}")
