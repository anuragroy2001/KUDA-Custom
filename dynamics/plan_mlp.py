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
from dynamics.dyn_utils import load_config_and_ckpt
from dynamics.mlp.mlp import MLP
from dynamics.tracker.spatracker_wrapper import SpaTrackerWrapper
from dynamics.video_recorder import VideoRecorder
from utils import set_seed, truncate_points

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIM_REAL_RATIO = 1000.
PUSH_LENGTH = 10 # length in simulation
T_SCALE = 1.0

# ============ action format ============
def decode_action(action, push_length=10):
    x_start = action[..., 0]
    y_start = action[..., 1]
    theta = action[..., 2]
    length = action[..., 3]
    action_repeat = length.to(torch.int32)
    x_end = x_start + push_length * torch.cos(theta)
    y_end = y_start + push_length * torch.sin(theta)
    decoded_action = torch.stack([x_start, y_start, x_end, y_end], dim=-1)
    return decoded_action, action_repeat

# for closed-loop control
def step_and_record(env, action, video_recorder):
    # decode to full action
    x_start = action[0]
    y_start = action[1]
    theta = action[2]
    length = action[3]
    action_repeat = int(length)
    x_end = x_start + action_repeat * PUSH_LENGTH * math.cos(theta)
    y_end = y_start + action_repeat * PUSH_LENGTH * math.sin(theta)
    decoded_action = np.array([x_start, y_start, x_end, y_end])

    decoded_action = decoded_action_sim_to_real(decoded_action)
    x_start, y_start, x_end, y_end = decoded_action

    # rope
    # yaw = None
    # pusher
    yaw = theta

    env.reset_to_default_pose()
    time.sleep(0.5)
    video_recorder.start()
    env.move_to_table_position(x_start, y_start, 0.1, yaw)
    env.move_to_table_position(x_start, y_start, 0.0, yaw)
    time.sleep(0.5)
    env.move_to_table_position(x_end, y_end, 0.0, yaw)
    env.move_to_table_position(x_end, y_end, 0.1, yaw)
    time.sleep(0.5)
    env.reset_to_default_pose()
    if video_recorder.is_alive():
        video_recorder.terminate()
        video_recorder.join()

# ============ coordinate transformation ============
# using an extended z-axis in pymunk, but only using x and y when dynamics

def decoded_action_real_to_sim(action):
    """turn real action into simulator coordinate"""
    action_sim = deepcopy(action)
    action_sim *= SIM_REAL_RATIO
    return action_sim

def decoded_action_sim_to_real(action_sim):
    """turn action in simulator coordinate into real"""
    action_real = deepcopy(action_sim)
    action_real /= SIM_REAL_RATIO
    return action_real

def particle_real_to_sim(particles):
    if isinstance(particles, list):
        particles = np.array(particles).astype(np.float32)
    particles_sim = particles * SIM_REAL_RATIO
    return particles_sim

def particle_sim_to_real(particles_sim):
    if isinstance(particles_sim, list):
        particles_sim = np.array(particles_sim).astype(np.float32)
    particles_real = particles_sim / SIM_REAL_RATIO
    return particles_real

def preprocess(bsz_points):
    # bsz_points: (B, n_his=1, dim)
    return bsz_points / (512 * T_SCALE) * 2 - 1

def postprocess(bsz_points):
    # bsz_points: (B, -1)
    return (bsz_points + 1) / 2 * (512 * T_SCALE)

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
    return torch.mean(torch.norm(torch.add(x_target[..., :2], -target[..., :2]), 2, dim=2) ** 2, dim=1)

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
        
        x_ends = xs + lengths * PUSH_LENGTH * torch.cos(thetas)
        y_ends = ys + lengths * PUSH_LENGTH * torch.sin(thetas)

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

            thetas_i = torch.atan2(y_ends_i - ys_i, x_ends_i - xs_i)
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
    x_ends = xs + lengths * PUSH_LENGTH * torch.cos(thetas)
    y_ends = ys + lengths * PUSH_LENGTH * torch.sin(thetas)

    x = torch.sum(weight_seqs * xs, dim=0)  # (n_look_ahead,)
    y = torch.sum(weight_seqs * ys, dim=0)  # (n_look_ahead,)
    x_end = torch.sum(weight_seqs * x_ends, dim=0)  # (n_look_ahead,)
    y_end = torch.sum(weight_seqs * y_ends, dim=0)  # (n_look_ahead,)

    theta = torch.atan2(y_end - y, x_end - x)  # (n_look_ahead,)
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

# TODO
def running_cost(state, action, state_cur, material, target_specification=None, target_box=None, weights=None):
    # chamfer distance
    # state: (bsz, n_look_forward, 10)
    # action: (bsz, n_look_forward, action_dim)
    # target_specification: (target_indices (n,), target_state (n, 3))
    # state_cur: (8)
    # weights: (n_look_forward,)
    bsz = state.shape[0]
    n_look_forward = state.shape[1]

    # reshape state and state_cur to point form
    state = state[:, :, :8].reshape(bsz, n_look_forward, 4, 2)
    state_flat = state.reshape(bsz * n_look_forward, 4, 2)
    state_cur = state_cur[:8].reshape(4, 2)

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

    # elif target_box is not None:
    #     chamfer_distance = box_loss(state_flat, target_box).reshape(bsz, n_look_forward)
    
    distance_weight = 2. / (distance.max().item() + 1e-6)
 
    if material == 'T_shape':
        x_start = action[:, :, 0]  # (bsz, n_look_forward)
        y_start = action[:, :, 1]  # (bsz, n_look_forward)
        theta = action[:, :, 2]
        action_point_2d = torch.stack([x_start, y_start], dim=-1)
        action_point_2d = action_point_2d.reshape(bsz, n_look_forward, 1, 2)  # (bsz, n_look_forward, 1, 2)
        x_max = state_cur[:, 0].max(dim=0).values + 0.02 * SIM_REAL_RATIO
        x_min = state_cur[:, 0].min(dim=0).values - 0.02 * SIM_REAL_RATIO
        y_max = state_cur[:, 1].max(dim=0).values + 0.02 * SIM_REAL_RATIO
        y_min = state_cur[:, 1].min(dim=0).values - 0.02 * SIM_REAL_RATIO
        collision_penalty = torch.stack([
            torch.maximum(x_min - action_point_2d[:, :, :, 0], torch.zeros_like(x_min)),
            torch.maximum(action_point_2d[:, :, :, 0] - x_max, torch.zeros_like(x_max)),
            torch.maximum(y_min - action_point_2d[:, :, :, 1], torch.zeros_like(y_min)),
            torch.maximum(action_point_2d[:, :, :, 1] - y_max, torch.zeros_like(y_max)),
        ], dim=-1)
        collision_penalty = collision_penalty.max(dim=-1).values.min(dim=-1).values  # (bsz, n_look_forward)
        collision_penalty = torch.exp(-collision_penalty * 1000.)  # (bsz, n_look_forward)
    else:
        raise NotImplementedError(f"material {material} not implemented")

    # bbox in simulator
    bbox = np.array([[0.3, 0.6], [-0.3, 0.3]]) * SIM_REAL_RATIO
    xmax = state.max(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    xmin = state.min(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    ymax = state.max(dim=2).values[:, :, 1]  # (bsz, n_look_forward)
    ymin = state.min(dim=2).values[:, :, 1]  # (bsz, n_look_forward)

    box_penalty = torch.stack([
        torch.maximum(xmin - bbox[0, 0], torch.zeros_like(xmin)),
        torch.maximum(bbox[0, 1] - xmax, torch.zeros_like(xmax)),
        torch.maximum(ymin - bbox[1, 0], torch.zeros_like(ymin)),
        torch.maximum(bbox[1, 1] - ymax, torch.zeros_like(ymax)),
    ], dim=-1)  # (bsz, n_look_forward, 4)
    box_penalty = torch.exp(-box_penalty * 1000.).max(dim=-1).values  # (bsz, n_look_forward)

    reward = -distance_weight * distance - 50. * collision_penalty.mean(dim=1) - 5. * box_penalty.mean(dim=1)  # (bsz,)

    print(f'min distance {distance.min().item()}, max reward {reward.max().item()}')
    out = {
        "reward_seqs": reward,
    }
    return out

# ============ dynamics ============
@torch.no_grad()
def dynamics(state, perturbed_action, model, material, device):
    # state: (8)
    time0 = time.time()
    bsz = perturbed_action.shape[0]
    n_look_forward = perturbed_action.shape[1]
    n_his = 1

    decoded_action, action_repeat = decode_action(perturbed_action, push_length=PUSH_LENGTH)

    # use bsz_state, input_action as model input, and update them
    bsz_state = state[None, None].repeat(bsz, n_his, 1) # (bsz, n_his, 8)
    # add start pusher position
    bsz_state = torch.cat([bsz_state, decoded_action[:, 0, :2][:, None, :]], dim=-1)
    
    pred_state_seq = torch.zeros((bsz, n_look_forward, 10), device=device)

    for li in range(n_look_forward):
        print(f"look forward iter {li}")

        if li > 0:
            bsz_state = pred_state_seq[:, li-1:li].detach().clone().repeat(bsz, n_his, 1)

        if material == 'T_shape':
            input_action = decoded_action[:, li:li+1, 2:4]

        for ai in range(1, 1 + action_repeat[:, li].max().item()):
            pred_state = model(preprocess(bsz_state), preprocess(input_action)) # pred_state: (B, -1)
            pred_state = postprocess(pred_state)

            repeat_mask = (action_repeat[:, li] == ai)
            pred_state_seq[repeat_mask, li] = pred_state[repeat_mask].clone()

            bsz_state = torch.cat([pred_state[:, None, :8], input_action], dim=-1)
            input_action = input_action + (decoded_action[:, li:li+1, 2:4] - decoded_action[:, li:li+1, 0:2])

    out = {
        "state_seqs": pred_state_seq, # (bsz, n_look_forward, 10)
        "action_seqs": decoded_action, # (bsz, n_look_forward, 4)
    }
    time1 = time.time()
    print(f"model time {time1 - time0}")
    return out

def dynamics_chunk(state, perturbed_action, model, material, device, n_sample_chunk):
    # state: (8)
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
    
    # turn state into point form
    state = state[:8].reshape(4, 2)
    # add z axis
    state = torch.cat([state, 10 * torch.ones((4, 1), device=state.device)], dim=-1)

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
    # state_init: (10)
    # state: (n_look_forward, 10)
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
    state_best = res['best_model_output']['state_seqs'][0]  # (n_look_forward, 10)

    # plot
    action, repeat = decode_action(action_best.unsqueeze(0), push_length=PUSH_LENGTH)  # (1, n_look_forward, action_dim)
    action = action[0]  # (n_look_forward, action_dim)
    repeat = repeat[0, 0].item()

    state_init_vis = state_init.detach().cpu().numpy()  # (10)
    state_vis = state_best[0].detach().cpu().numpy()  # (10)
    action_vis = action[0].detach().cpu().numpy()  # (action_dim,)
    # turn into point form
    state_init_vis = state_init_vis[:8].reshape(4, 2)
    state_init_vis = np.concatenate([state_init_vis, 10 * np.ones((4, 1))], axis=-1)
    state_vis = state_vis[:8].reshape(4, 2)
    state_vis = np.concatenate([state_vis, 10 * np.ones((4, 1))], axis=-1)

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

    # action arrow
    x_start = action_vis[0]
    y_start = action_vis[1]
    x_end = action_vis[2]
    y_end = action_vis[3]
    x_delta = x_end - x_start
    y_delta = y_end - y_start
    z = 10
    if material == 'cloth':
        z += 0.01 * SIM_REAL_RATIO
    arrow_size = 3
    tip_length = 0.5
    for i in range(repeat):
        action_start_point = np.array([x_start + i * x_delta, y_start + i * y_delta, z])
        action_end_point = np.array([x_end + i * x_delta, y_end + i * y_delta, z])
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
    
    rgb_vis = cv2.addWeighted(rgb_overlay, 0.5, rgb_vis, 0.5, 0)
    
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'rgb_vis_{postfix}.png'), rgb_vis)
        cv2.imwrite(os.path.join(save_dir, f'rgb_orig_{postfix}.png'), rgb_orig)

# for debug
def visualize_push(state_init, res, model, rgb_vis, material, intr, extr, device, save_dir=None, postfix=None):
    # state_init: (10)
    # rgb_vis: np.ndarray (H, W, 3)
    n_his = 1

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

    bsz_state = state_init[None, None].repeat(1, n_his, 1)
    bsz_state = torch.cat([bsz_state, decoded_action[:, 0:1, :2]], dim=-1)
    input_action = decoded_action[:, 0:1, 2:4]

    # draw the current state
    states_vis = state_init[:8].reshape(4, 2)
    states_vis = torch.cat([states_vis, 10 * torch.ones((4, 1), device=states_vis.device)], dim=-1)
    states_vis = states_vis.detach().cpu().numpy()
    states_proj = project(states_vis, intr, extr)
    rgb_tem = rgb_vis.copy()
    for k in range(states_proj.shape[0]):
        cv2.circle(rgb_tem, (int(states_proj[k, 0]), int(states_proj[k, 1])), point_size, color, -1)
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'push_vis_{postfix}_0.png'), rgb_tem)

    # rollout and visualize
    for ai in range(1, 1 + action_repeat[:, 0].max().item()):
        pred_state = model(preprocess(bsz_state), preprocess(input_action)) # pred_state: (B, -1)
        pred_state = postprocess(pred_state)

        # visualize
        states_cur_vis = pred_state[0, :8].reshape(4, 2)
        states_cur_vis = torch.cat([states_cur_vis, 10 * torch.ones((4, 1), device=states_cur_vis.device)], dim=-1)
        states_cur_vis = states_cur_vis.detach().cpu().numpy()
        states_cur_proj = project(states_cur_vis, intr, extr)
        rgb_tem = rgb_vis.copy()
        for k in range(states_cur_proj.shape[0]):
            cv2.circle(rgb_tem, (int(states_cur_proj[k, 0]), int(states_cur_proj[k, 1])), point_size, color, -1)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, f'push_vis_{postfix}_{ai}.png'), rgb_tem)

        bsz_state = torch.cat([pred_state[:, None, :8], input_action], dim=-1)
        input_action = input_action + (decoded_action[:, 0:1, 2:4] - decoded_action[:, 0:1, :2])

# ============ state ============
def get_state_cur(object, env, device, top_down_cam=0, visualize=False):
    # not the real state, lack of pusher position
    object_pcds_list = [env.get_points_by_name(object, camera_index=top_down_cam, debug=visualize)]
    object_pcds = [item for sublist in object_pcds_list for item in sublist]
    obj_kps = np.concatenate(object_pcds, axis=0) # (4, 3)
    state_cur = obj_kps[:, :2].reshape(-1) # (8)
    state_cur = particle_real_to_sim(state_cur)
    state_cur = torch.tensor(state_cur, device=device, dtype=torch.float32)

    rgb_vis = env.get_rgb_depth_pc()[0][top_down_cam]

    return state_cur, obj_kps, rgb_vis

# ============ MOTION PLANNING ============
def closed_loop_plan(
        object_points,
        target_specification,
        object,
        material,
        env,
        top_down_cam,
        tracking_cam,
        save_dir,
        track_as_state=False,
        target_pcd=None,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bbox = env.get_bbox()[:2]
    bbox[:, 0] += 0.05
    bbox[:, 1] -= 0.05

    # transfer bbox into simulator
    if material == "T_shape":
        bbox *= SIM_REAL_RATIO
        action_lower_lim = [
            bbox[0, 0],
            bbox[1, 0],
            -math.pi,
            3,
        ]
        action_upper_lim = [
            bbox[0, 1],
            bbox[1, 1],
            math.pi,
            20,
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

    model = MLP(
        obs_dim=model_config['obs_dim'],
        action_dim=model_config['action_dim'],
        n_history=model_config['n_history'],
        architecture=model_config['architecture'],
        block_center=True,
    )
    model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    tracker = SpaTrackerWrapper()

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

    for i in range(n_actions):
        time1 = time.time()

        # get state
        if i == 0:
            state_cur = particle_real_to_sim(object_points)
            state_cur = state_cur[:, :2].reshape(-1)
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
        # use this to visualize the push process
        # visualize_push(state_cur, res, model, obs, material, intr, extr, device, save_dir=save_dir, postfix=f'{i}_0')
        print("start action")

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
            object_pcd = env.get_points_by_name(material, camera_index=top_down_cam, fuse=True)
            object_pcd = np.asarray(object_pcd)
            chamfer_dist = chamfer(target_pcd, object_pcd)[0]
            print(f"chamfer distance: {chamfer_dist}")
            # log
            with open(f'{save_dir}/chamfer_log.txt', 'a') as f:
                f.write(f"chamfer distance: {chamfer_dist}\n")


        # tracking
        if i == 0:
            queries = object_points
        else:
            queries = obj_kps
        queries = env.world_to_viewport(queries, tracking_cam)
        video_dir = f'{save_dir}/tracking_{i}'
        frames = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        video = []
        for frame in frames:
            img = Image.open(frame)
            img = np.array(img)
            video.append(img)
        video = np.stack(video, axis=0)
        depths = []
        depth_files = sorted(glob.glob(os.path.join(video_dir, '*.png')))
        for depth_file in depth_files:
            depth = Image.open(depth_file)
            depth = np.array(depth) / 1000.
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        video = video[:len(depths)] # sometimes the last depth frame is not saved
        pred_tracks, pred_visibility = tracker(video, depths, queries, debug=True, save_dir=video_dir)

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

        # vis
        obs = env.get_rgb_depth_pc()[0][top_down_cam]
        vis_img = view_specification(obs, state_cur, target_specification, intr, extr)
        cv2.imwrite(os.path.join(save_dir, f'rgb_original_{i}_1.png'), obs)
        visualize_img(state_cur, res, vis_img, material, intr, extr,
                    save_dir=save_dir, postfix=f'{i}_1')

        time2 = time.time()
        print(f"step {i} time {time2 - time1}")
