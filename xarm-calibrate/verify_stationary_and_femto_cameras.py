import time
import pickle
from typing import Union, Optional, Any
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from pyorbbecsdk import Config, Context, OBSensorType, OBFormat, Pipeline, FrameSet, VideoFrame, OBLogLevel, AlignFilter, OBStreamType

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame
            return frame
        result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class FemtoEnv:
    def __init__(self, vis_dir='calibration_result_femto'):
        self.vis_dir = vis_dir
        self.calibrate_result_dir = vis_dir
        self.setup_femto()

    def setup_femto(self):
        config = Config()
        context = Context()
        # context.set_logger_level(OBLogLevel.NONE)
        self.pipeline = Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # color_profile = profile_list.get_video_stream_profile(3840, 2160, OBFormat.RGB, 5)
        color_profile = profile_list.get_video_stream_profile(1280, 960, OBFormat.RGB, 5)
        config.enable_stream(color_profile)
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_video_stream_profile(640, 576, OBFormat.Y16, 5)
        config.enable_stream(depth_profile)
        self.pipeline.start(config)
        time.sleep(5)
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.temporal_filter = TemporalFilter(alpha=0.5)

    def get_obs(self):
        frames = None
        while frames is None:
            frames = self.pipeline.wait_for_frames(200)
            frames = self.align_filter.process(frames)
        color_frame = frames.get_color_frame()
        color_image = frame_to_bgr_image(color_frame)

        depth_frame = frames.get_depth_frame()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        depth_data = self.temporal_filter.process(depth_data)
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return {'color': color_image, 'depth': depth_image}

    def get_intrinsics(self):
        camera_param = self.pipeline.get_camera_param()

        # Get RGB intrinsics
        intr = camera_param.rgb_intrinsic
        rgb_mat = np.eye(3)
        rgb_mat[0, 0] = intr.fx
        rgb_mat[1, 1] = intr.fy
        rgb_mat[0, 2] = intr.cx
        rgb_mat[1, 2] = intr.cy

        # Get depth intrinsics
        intr = camera_param.depth_intrinsic
        depth_mat = np.eye(3)
        depth_mat[0, 0] = intr.fx
        depth_mat[1, 1] = intr.fy
        depth_mat[0, 2] = intr.cx
        depth_mat[1, 2] = intr.cy
        return {'rgb': rgb_mat, 'depth': depth_mat}

    def get_femto_point_cloud(self, transform):
        intrinsics = self.get_intrinsics()
        depth_intrinsics = intrinsics['depth']
        obs = self.get_obs()
        color_image, depth_image = obs['color'], obs['depth']

        # Resize the depth image to match the resolution of the color image
        h_color, w_color = color_image.shape[:2]
        depth_image_resized = cv2.resize(depth_image, (w_color, h_color), interpolation=cv2.INTER_LINEAR)

        # Adjust depth intrinsics to match the resized depth image
        scale_x = w_color / depth_intrinsics[0, 2]
        scale_y = h_color / depth_intrinsics[1, 2]
        fx_resized = depth_intrinsics[0, 0] * scale_x
        fy_resized = depth_intrinsics[1, 1] * scale_y
        cx_resized = depth_intrinsics[0, 2] * scale_x
        cy_resized = depth_intrinsics[1, 2] * scale_y

        h, w = depth_image_resized.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth_image_resized
        X = (X - cx_resized) * Z / fx_resized
        Y = (Y - cy_resized) * Z / fy_resized

        # Filter based on valid depth values
        mask = Z > 0
        X, Y, Z = X[mask], Y[mask], Z[mask]
        points = np.vstack((X, Y, Z)).T

        # Apply transformation to the points
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_transformed_homogeneous = np.dot(points_homogeneous, transform.T)
        points_transformed = points_transformed_homogeneous[:, :3]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_transformed)

        # Get corresponding colors and apply to the point cloud
        colors = color_image[mask].reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        return pcd
    
    def close(self):
        self.pipeline.stop()


# Load existing camera calibration data
camera_to_bases = pickle.load(open('real_world/calibration_result/camera_to_bases.pkl', 'rb'))
serial_numbers = list(camera_to_bases.keys())
pcds = []

# Process RealSense cameras
for serial_number in serial_numbers:
    # if serial_number == '213622252214':
    #     continue
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    aligned_depth_frame = rs.align(rs.stream.color).process(frames).get_depth_frame()
    aligned_color_frame = frames.get_color_frame()
    pc = rs.pointcloud()
    pc.map_to(aligned_color_frame)
    points = pc.calculate(aligned_depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
    vtx_filtered = vtx[vtx[:, 2] <= 1.5]
    color_data = np.asanyarray(aligned_color_frame.get_data()).reshape(-1, 3)
    color_filtered = color_data[vtx[:, 2] <= 1.5]

    vtx_homogeneous = np.hstack((vtx_filtered, np.ones((vtx_filtered.shape[0], 1))))
    vtx_transformed_homogeneous = np.dot(vtx_homogeneous, camera_to_bases[serial_number].T)
    vtx_transformed = vtx_transformed_homogeneous[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx_transformed)
    pcd.colors = o3d.utility.Vector3dVector(color_filtered / 255.0)
    pcds.append(pcd)

# Add Femto camera
femto_env = FemtoEnv()
femto_intrinsics = femto_env.get_intrinsics()
femto_transform = pickle.load(open('calibration_result_femto/camera_to_bases.pkl', 'rb'))['femto']

femto_pcd = femto_env.get_femto_point_cloud(femto_transform)
femto_env.close()
pcds.append(femto_pcd)

# Merge all point clouds
merged_pcd = o3d.geometry.PointCloud()
for pcd in pcds:
    merged_pcd.points.extend(pcd.points)
    merged_pcd.colors.extend(pcd.colors)

# Filter points
points = np.asarray(merged_pcd.points)
colors = np.asarray(merged_pcd.colors)
mask = (points[:, 2] >= -0.1) & (points[:, 2] <= 0.2) & (points[:, 0] >= 0.1) & (points[:, 0] <= 0.9) & (points[:, 1] >= -0.55) & (points[:, 1] <= 0.45)
filtered_points = points[mask]
filtered_colors = colors[mask]

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(filtered_points)
merged_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Save and visualize the merged point cloud
# o3d.io.write_point_cloud('combined_pcd.pcd', merged_pcd)
o3d.visualization.draw_geometries([merged_pcd, coordinate_frame])
