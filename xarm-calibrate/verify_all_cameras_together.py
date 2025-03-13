import math
import pickle
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import time
from transforms3d.euler import euler2mat

from xarm.wrapper import XArmAPI

camera_to_bases = pickle.load(open('real_world/calibration_result/camera_to_bases.pkl', 'rb'))
serial_numbers = list(camera_to_bases.keys())
del serial_numbers[0]
pcds = []

# Stationary cameras
for serial_number in serial_numbers:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    time.sleep(4)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    aligned_depth_frame = frames.get_depth_frame()
    aligned_color_frame = frames.get_color_frame()
    pipeline.stop()
    pc = rs.pointcloud()
    pc.map_to(aligned_color_frame)
    points = pc.calculate(aligned_depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
    vtx_filtered = vtx[vtx[:, 2] <= 1.5]
    color_data = np.asanyarray(aligned_color_frame.get_data())
    # color_filtered = color_data[vtx[:, 2] <= 1]
    color_data = color_data.reshape(-1, 3)  # colors
    color_filtered = color_data[vtx[:, 2] <= 1.5]

    vtx_homogeneous = np.hstack((vtx_filtered, np.ones((vtx_filtered.shape[0], 1))))
    vtx_transformed_homogeneous = np.dot(vtx_homogeneous, camera_to_bases[serial_number].T)
    vtx_transformed = vtx_transformed_homogeneous[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx_transformed)
    pcd.colors = o3d.utility.Vector3dVector(color_filtered / 255.0)  # Normalize colors to 0-1
    pcds.append(pcd)

# Wrist camera
serial_number = '246322303938'

hand_eye = pickle.load(open('real_world/calibration_result/calibration_handeye_result.pkl', 'rb'))
gripper_to_camera = np.concatenate((hand_eye['R_gripper2cam'], np.expand_dims(hand_eye['t_gripper2cam'], axis=1)), axis=1)
gripper_to_camera = np.concatenate((gripper_to_camera, np.array([[0, 0, 0, 1]])), axis=0)
camera_to_gripper = np.linalg.inv(gripper_to_camera)

robot = XArmAPI('192.168.1.228')
robot.motion_enable(enable=True)
robot.set_mode(0)
robot.set_state(state=0)

state = robot.get_position()[1]
position = [x / 1000. for x in state[:3]]
rpy = state[3:6]
rotation_matrix = euler2mat(math.radians(rpy[0]), math.radians(rpy[1]), math.radians(rpy[2]))
gripper_to_base = np.concatenate((rotation_matrix, np.expand_dims(position, axis=1)), axis=1)
gripper_to_base = np.concatenate((gripper_to_base, np.array([[0, 0, 0, 1]])), axis=0)

# camera_to_base = np.dot(np.linalg.inv(camera_to_gripper), gripper_to_base)
camera_to_base = np.dot(gripper_to_base, camera_to_gripper)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial_number)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
time.sleep(4)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
align = rs.align(rs.stream.color)
frames = align.process(frames)
aligned_depth_frame = frames.get_depth_frame()
aligned_color_frame = frames.get_color_frame()
pipeline.stop()
pc = rs.pointcloud()
pc.map_to(aligned_color_frame)
points = pc.calculate(aligned_depth_frame)
vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
vtx_filtered = vtx[vtx[:, 2] <= 1.5]
color_data = np.asanyarray(aligned_color_frame.get_data())
color_data = color_data.reshape(-1, 3)  # colors
color_filtered = color_data[vtx[:, 2] <= 1.5]

vtx_homogeneous = np.hstack((vtx_filtered, np.ones((vtx_filtered.shape[0], 1))))
vtx_transformed_homogeneous = np.dot(vtx_homogeneous, camera_to_base.T)
# vtx_transformed_homogeneous = np.dot(vtx_homogeneous, np.eye(4))
vtx_transformed = vtx_transformed_homogeneous[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vtx_transformed)
pcd.colors = o3d.utility.Vector3dVector(color_filtered / 255.0)  # Normalize colors to 0-1
pcds.append(pcd)

merged_pcd = o3d.geometry.PointCloud()
for pcd in pcds:
    merged_pcd.points.extend(pcd.points)
    merged_pcd.colors.extend(pcd.colors)

# Filter points
points = np.asarray(merged_pcd.points)
colors = np.asarray(merged_pcd.colors)
mask = (points[:, 2] >= -0.1) & (points[:, 2] <= 0.5) & (points[:, 0] >= 0.17) & (points[:, 0] <= 0.9) & (points[:, 1] >= -0.40) & (points[:, 1] <= 0.55)
filtered_points = points[mask]
filtered_colors = colors[mask]

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(filtered_points)
merged_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.io.write_point_cloud('real_pcd.pcd', merged_pcd)
o3d.visualization.draw_geometries([merged_pcd, coordinate_frame])
