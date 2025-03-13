import pickle
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2

serial_number = '311322303615'
rvecs = pickle.load(open('real_world/calibration_result/rvecs.pkl', 'rb'))
tvecs = pickle.load(open('real_world/calibration_result/tvecs.pkl', 'rb'))
hand_eye = pickle.load(open('real_world/calibration_result/calibration_handeye_result.pkl', 'rb'))
rvec = rvecs[serial_number]
tvec = np.squeeze(tvecs[serial_number])

R_mtx, _ = cv2.Rodrigues(rvec)
world_to_camera = np.array([[R_mtx[0][0], R_mtx[0][1], R_mtx[0][2], tvec[0]],
                  [R_mtx[1][0], R_mtx[1][1], R_mtx[1][2], tvec[1]],
                  [R_mtx[2][0], R_mtx[2][1], R_mtx[2][2], tvec[2]],
                  [0, 0, 0, 1]])
camera_to_world = np.linalg.inv(world_to_camera)

# np.expand_dims(arr, axis=1)
base_to_world = np.concatenate((hand_eye['R_base2world'], np.expand_dims(hand_eye['t_base2world'], axis=1)), axis=1)
base_to_world = np.concatenate((base_to_world, np.array([[0, 0, 0, 1]])), axis=0)


camera_to_base = np.dot(np.linalg.inv(base_to_world), camera_to_world)
# camera_to_world = np.dot(camera_to_world, base_to_world)


pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial_number)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Align the depth frame to color frame
align = rs.align(rs.stream.color)
frames = align.process(frames)

# Get aligned frames
aligned_depth_frame = frames.get_depth_frame()
aligned_color_frame = frames.get_color_frame()

pipeline.stop()

pc = rs.pointcloud()
pc.map_to(aligned_color_frame)

# Generate the pointcloud and texture mappings
points = pc.calculate(aligned_depth_frame)

vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
vtx_filtered = vtx[vtx[:, 2] <= 1]

# Get color data
color_data = np.asanyarray(aligned_color_frame.get_data())
color_data = np.reshape(color_data, (480, 640, 3))
color_data = color_data.reshape(-1, 3)  # colors
color_filtered = color_data[vtx[:, 2] <= 1]


vtx_homogeneous = np.hstack((vtx_filtered, np.ones((vtx_filtered.shape[0], 1))))
vtx_transformed_homogeneous = np.dot(vtx_homogeneous, camera_to_base.T)
vtx_transformed = vtx_transformed_homogeneous[:, :3]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vtx_transformed)
pcd.colors = o3d.utility.Vector3dVector(color_filtered / 255.0)  # Normalize colors to 0-1

# Visualize the point cloud
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, coordinate_frame])
###############



