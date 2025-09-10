import pyrealsense2 as rs
import time
import math
import pickle
import torch
import numpy as np
from copy import deepcopy
import open3d as o3d
import cv2
# from xarm.wrapper import XArmAPI
from transforms3d.euler import quat2euler, euler2mat
from utils import pc2voxel, voxel2index

from multiprocessing.managers import SharedMemoryManager
from envs.real_world.camera.multi_realsense import MultiRealsense, SingleRealsense
from perception.predictor import GroundingSegmentPredictor
from scipy.spatial.transform import Rotation as R
import socket

# Important!!! Check before experiment
# EE_LENGTH = 0.173 # stick
# EE_LENGTH = 0.104 # pusher
EE_LENGTH = 0.0 
Z_PUSHER = -0.062 # the lowest position of the pusher
# scale between T mesh and the real T
T_SCALE = 1.0


class RealEnv:
    def __init__(self, env_config=None):
        self.config = env_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WH = env_config.WH
        self.capture_fps = env_config.capture_fps
        self.obs_fps = env_config.obs_fps
        self.n_obs_steps = env_config.n_obs_steps
        self.sample_points = env_config.num_points
        # workspace bounds
        self.workspace_bounds_min = np.array([-1.0, 0.0, 0.0])
        self.workspace_bounds_max = np.array([0.0, 1.0, 1.0])

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        print(f'Found {len(self.serial_numbers)} fixed cameras.')

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense =  MultiRealsense(
                serial_numbers=self.serial_numbers,
                shm_manager=self.shm_manager,
                resolution=(self.WH[0], self.WH[1]),
                capture_fps=self.capture_fps,
                enable_color=True,
                enable_depth=True,
                process_depth=env_config.process_depth,
                get_max_k=self.capture_fps,
                verbose=env_config.verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance()
        self.last_realsense_data = None
        self.use_robot = env_config.use_robot

        if self.use_robot:
            self.action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.action_sock.connect(('localhost', 5000))

            self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.status_sock.connect(('localhost', 50005))
            self.initialize_robot()

        self.setup_cameras()

        start_time = time.time()
        self.predictor = GroundingSegmentPredictor(show_bbox=False, show_mask=False)
        end_time = time.time()
        print(f'GroundingSegmentPredictor initialized in {end_time - start_time:.4f} seconds')

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_bbox(self):
        return np.array([self.workspace_bounds_min, self.workspace_bounds_max]).T

    def start_recording(self, file_path, start_time=None):
        self.realsense.start_recording(file_path, start_time=start_time)

    def initialize_robot(self):
        print('Initializing robot...')
        # self.robot = XArmAPI('192.168.1.209')
        # self.robot.motion_enable(enable=True)
        # self.robot.set_mode(0)
        # self.robot.set_state(state=0)
        # self.robot_default_pos = [0, -60, -30, 0, 90, 0]
        self.reset_to_default_pose()
        # self.recent_target_speed = 0

    def setup_cameras(self):
        # To get good point cloud
        self.camera_depth_to_disparity = rs.disparity_transform(True)
        self.camera_disparity_to_depth = rs.disparity_transform(False)
        self.camera_spatial = rs.spatial_filter()
        self.camera_spatial.set_option(rs.option.filter_magnitude, 5)
        self.camera_spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_spatial.set_option(rs.option.filter_smooth_delta, 1)
        self.camera_spatial.set_option(rs.option.holes_fill, 1)
        self.camera_temporal = rs.temporal_filter()
        self.camera_temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_temporal.set_option(rs.option.filter_smooth_delta, 1)
        self.camera_threshold = rs.threshold_filter()
        self.camera_threshold.set_option(rs.option.min_distance, 0) # Minimum distance in meters
        self.camera_threshold.set_option(rs.option.max_distance, 3) # Maximum distance in meters (3m)

        # camera_to_robot_transforms = pickle.load(open('xarm-calibrate/real_world/calibration_result/camera_to_bases.pkl', 'rb'))
        # self.camera_to_robot_transforms = []
        # for serial in self.serial_numbers:
        #     self.camera_to_robot_transforms.append(camera_to_robot_transforms[serial])
        # self.camera_to_robot_transforms = []
        # t = np.array([-0.65929,  0.72136,  0.85829], dtype=np.float32)
        # q = np.array([-0.37775, 0.91128, -0.14943, 0.06750], dtype=np.float32)
        # R_mat = R.from_quat(q).as_matrix().astype(np.float32)

        # T = np.eye(4, dtype=np.float32)
        # T[:3, :3] = R_mat
        # T[:3,  3] = t

        # # self.camera_to_robot_transforms = [T]
        # # if self.serial_numbers has two entries, this makes two copies of T:
        # self.camera_to_robot_transforms = [T.copy() for _ in self.serial_numbers]
        # for serial, transform in zip(self.serial_numbers, self.camera_to_robot_transforms):
        #     print(f"Camera serial: {serial}, Transform:\n{transform}")
        calibrations = {
        "151422253661": (  # replace with your first camera’s serial
            np.array([-0.49387,  0.59460,  0.89782], dtype=np.float32),
            np.array([-0.37936, 0.92398, -0.04728, 0.01018], dtype=np.float32)
        ),
        "151422254605": (  # replace with your second camera’s serial
            np.array([ -0.99265, 0.72944,  0.35029], dtype=np.float32),
            np.array([ -0.35501,  0.84646,  -0.37527, 0.12901], dtype=np.float32)
        ),
        # add more cameras here if needed…
    }

        self.camera_to_robot_transforms = []
        for serial in self.serial_numbers:
            if serial not in calibrations:
                raise KeyError(f"No calibration found for camera {serial}")
            t, q = calibrations[serial]
            R_mat = R.from_quat(q).as_matrix().astype(np.float32)

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_mat
            T[:3,  3] = t
            self.camera_to_robot_transforms.append(T)

            print(f"Camera serial: {serial}, Transform:\n{T}")


    def get_intrinsic(self, camera_index=0):
        return self.realsense.get_intrinsics()[camera_index]
    
    def get_extrinsic(self, camera_index=0):
        return self.camera_to_robot_transforms[camera_index]

    def reset(self):
        self.initialize_robot()

    def get_rgb_depth_pc(self, return_pcd=False, camera_index=None):
        assert self.is_ready

        # get data
        k = math.ceil(self.capture_fps / self.obs_fps)
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        rgb_images, rgb_data, depths, pcds = [], [], [], []
        for camera_idx, value in self.last_realsense_data.items():
            if camera_index is not None and camera_idx != camera_index:
                continue
            rgb_images.append(value['color'][-1])
            depths.append(value['depth'][-1])
            
            if return_pcd:
                depth = value['depth'][-1]
                depth = depth / 1000
                mask = np.ones_like(depth, dtype=bool)
                vtx = self._depth2fgpcds(camera_idx, depth, [mask])[0]
                color_data = value['color'][-1]
                color_data = np.reshape(color_data, (self.WH[1], self.WH[0], 3))
                mask = np.logical_and(mask, depth > 0)
                color_data = color_data[mask]
                color_data = color_data.reshape(-1, 3)  # colors
                color_data = color_data[:, ::-1]  # BGR to RGB

                vtx_homogeneous = np.hstack((vtx, np.ones((vtx.shape[0], 1))))
                vtx_transformed_homogeneous = np.dot(vtx_homogeneous, self.camera_to_robot_transforms[camera_idx].T)
                vtx_transformed = vtx_transformed_homogeneous[:, :3]

                # Exclude points
                indices = vtx_transformed[:, 2] >= -0.1 # Exclude points that are low
                vtx_transformed = vtx_transformed[indices]
                color_data = color_data[indices]

                target_num_points = 150000
                indices = np.random.choice(np.asarray(vtx_transformed).shape[0], target_num_points, replace=False)
                sampled_vtx_transformed = np.asarray(vtx_transformed)[indices, :]
                sampled_color_data = color_data[indices, :]

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sampled_vtx_transformed)
                pcd.colors = o3d.utility.Vector3dVector(sampled_color_data / 255.0)  # Normalize colors to 0-1
                pcds.append(pcd)

        if return_pcd:
            merged_pcd = o3d.geometry.PointCloud()
            for pcd in pcds:
                merged_pcd.points.extend(pcd.points)
                merged_pcd.colors.extend(pcd.colors)
            merged_pcd.estimate_normals()
            # clip pcd
            merged_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=self.workspace_bounds_min, max_bound=self.workspace_bounds_max))
            return rgb_images, rgb_data, depths, merged_pcd
        else:
            return rgb_images, rgb_data, depths, None

    def _depth2fgpcds(self, i, depth, masks):
        # depth: (h, w)
        # fgpcds: (m, n, 3)
        # masks: (m, h, w)
        h, w = depth.shape
        camera_intrinsic = self.get_intrinsic(i)
        fgpcds = []
        if masks is None:
            return fgpcds
        for i in range(len(masks)):
            mask = masks[i]
            mask = np.logical_and(mask, depth > 0)
            fgpcd = np.zeros((mask.sum(), 3))
            pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
            pos_x = pos_x[mask]
            pos_y = pos_y[mask]
            points = np.stack([pos_x, pos_y, np.ones_like(pos_x)], axis=1)
            points = points * depth[mask][:, None]
            points = points @ np.linalg.inv(camera_intrinsic).T
            fgpcds.append(points)
        return fgpcds

    # def _merge_pcd(self, pcds, dist_threshold=0.01):
    #     """
    #     Merge point clouds by distance.
    #     """
    #     merged_pcds = []
    #     for pcd in pcds:
    #         min_dist = 1e9
    #         min_dist_pcd = None
    #         for merged_pcd in merged_pcds:
    #             dist = self._get_dist(pcd, merged_pcd)
    #             if dist < dist_threshold and dist < min_dist:
    #                 min_dist = dist
    #                 min_dist_pcd = merged_pcd
    #         if min_dist_pcd is not None:
    #             min_dist_pcd.points.extend(pcd.points)
    #             min_dist_pcd.colors.extend(pcd.colors)
    #         else:
    #             merged_pcds.append(pcd)
    #     return merged_pcds

    # def _get_dist(self, pcd1, pcd2):
    #     pcd1 = np.asarray(pcd1.points)
    #     pcd2 = np.asarray(pcd2.points)
    #     return np.linalg.norm(np.mean(pcd1, axis=0) - np.mean(pcd2, axis=0))

    def _merge_pcd(self, pcds, iou_threshold=0.05):
        """
        Merge point clouds by intersection over union.
        """
        merged_pcds = []
        for pcd in pcds:
            max_iou = 0
            max_iou_pcd = None
            for merged_pcd in merged_pcds:
                iou = self._get_iou(pcd, merged_pcd)
                if iou > iou_threshold and iou > max_iou:
                    max_iou = iou
                    max_iou_pcd = merged_pcd
            if max_iou_pcd is not None:
                max_iou_pcd.points.extend(pcd.points)
                max_iou_pcd.colors.extend(pcd.colors)
            else:
                merged_pcds.append(pcd)
        return merged_pcds

    def _get_iou(self, pcd1, pcd2, map_size=30):
        voxels1 = pc2voxel(
            np.asarray(pcd1.points),
            np.array(self.workspace_bounds_min),
            np.array(self.workspace_bounds_max),
            map_size=map_size
        )
        indecies1 = voxel2index(voxels1, map_size=map_size)
        voxels2 = pc2voxel(
            np.asarray(pcd2.points),
            np.array(self.workspace_bounds_min),
            np.array(self.workspace_bounds_max),
            map_size=map_size
        )
        indecies2 = voxel2index(voxels2, map_size=map_size)
        intersection = len(
            set(indecies1).intersection(set(indecies2))
        )
        union = len(
            set(indecies1).union(set(indecies2))
        )
        return intersection / union

    def _get_transformed_object_pcd(self, depths, maskses, camera_index=None, debug=False):
        partial_pcds = []
        for i, (depth, masks) in enumerate(zip(depths, maskses)):
            depth = depth / 1000 # Convert to meters
            if camera_index is not None:
                assert len(depths) == 1
                i = camera_index
            obj_pcds = self._depth2fgpcds(i, depth, masks)
            for obj_pcd in obj_pcds:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(obj_pcd)
                
                # Convert Open3D PointCloud to NumPy array
                vtx = np.asarray(point_cloud.points)

                # Convert to homogeneous coordinates
                vtx_homogeneous = np.hstack((vtx, np.ones((vtx.shape[0], 1))))

                # Apply the transformation
                # Assuming self.camera_to_robot_transform is your transformation matrix
                vtx_transformed_homogeneous = np.dot(vtx_homogeneous, self.camera_to_robot_transforms[i].T)

                # Convert back to standard coordinates
                vtx_transformed = vtx_transformed_homogeneous[:, :3]

                # Update the points in the Open3D PointCloud
                point_cloud.points = o3d.utility.Vector3dVector(vtx_transformed)
                partial_pcds.append(point_cloud)

                # Visualize PointCloud
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([point_cloud, coordinate_frame])
                #Save as a .ply file to logs
                # o3d.io.write_point_cloud(f'logs/point_cloud_{i}.ply', point_cloud)
        # use merge if multiple cameras
        # merged_pcds = self._merge_pcd(partial_pcds)
        merged_pcds = partial_pcds
        # o3d.io.write_point_cloud(f'logs/merged_cloud_{i}.ply', merged_pcds[0])

        # filter the points by the workspace bounds
        for i in range(len(merged_pcds)):
            merged_pcds[i] = merged_pcds[i].crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=self.workspace_bounds_min, max_bound=self.workspace_bounds_max))
        
        if debug:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries(merged_pcds + [coordinate_frame])
        
        o3d.io.write_point_cloud(f'logs/merged_cloud_{i}.ply', merged_pcds[0])
        return merged_pcds

    def get_points_by_name(self, query_name, camera_index=None, return_normals=False, num_points=None, fuse=False, debug=False):
        """
        Get sampled points from the cameras queried by object name.
        """
        points_to_sample = self.sample_points if num_points is None else num_points
        query_name = query_name.lower()

        sampled_points, sampled_normals = [], []
        rgb_images, _, depths, _ = self.get_rgb_depth_pc(camera_index=camera_index)
        masks = self.predictor.predict(rgb_images, query_name)

        print("Visualizing masks for query:", query_name)
        obj_pcds = self._get_transformed_object_pcd(depths, masks, camera_index=camera_index, debug=False) # Visualize this point cloud https://chat.openai.com/share/b2405a32-3a8c-47f2-8cee-c387ef6ef5e4
        print(f'Found {len(obj_pcds)} objects for query: {query_name}')
        # Randomly sample 'num_points' points from the point cloud
        for i in range(len(obj_pcds)):
            if num_points is not None and len(obj_pcds[i].points) > num_points:
                sampled_indices = np.random.choice(len(obj_pcds[i].points), num_points, replace=False)
                obj_pcds[i] = obj_pcds[i].select_by_index(sampled_indices)

        # if fuse, simply fuse all pcds as one pcd
        if fuse:
            if return_normals:
                for obj_pcd in obj_pcds:
                    obj_pcd.estimate_normals()
                return [np.concatenate([np.asarray(obj_pcd.points) for obj_pcd in obj_pcds], axis=0)], [np.concatenate([np.asarray(obj_pcd.normals) for obj_pcd in obj_pcds], axis=0)]
            else:
                return [np.concatenate([np.asarray(obj_pcd.points) for obj_pcd in obj_pcds], axis=0)]

        # for T_shape, we use the keypoints to represent each object, is this correct for wire?
        if 't_shape' in query_name:
            obj_pcds = [self.T_postprocess(obj_pcd) for obj_pcd in obj_pcds]
            

        # for cubes, we use the center to represent each object
        if 'cube' in query_name:
            if return_normals:
                for obj_pcd in obj_pcds:
                    obj_pcd.estimate_normals()
                return [np.array([np.mean(np.asarray(obj_pcd.points), axis=0) for obj_pcd in obj_pcds])], [np.array([np.mean(np.asarray(obj_pcd.normals), axis=0) for obj_pcd in obj_pcds])]
            else:
                return [np.array([np.mean(np.asarray(obj_pcd.points), axis=0) for obj_pcd in obj_pcds])]

        # for coffee_beans, we simply fuse all pcds as one object
        if "coffee_beans" in query_name or "candy" in query_name or "cable" in query_name or 'wire' in query_name or 'rope' in query_name:
            if return_normals:
                for obj_pcd in obj_pcds:
                    obj_pcd.estimate_normals()
                return [np.concatenate([np.asarray(obj_pcd.points) for obj_pcd in obj_pcds], axis=0)], [np.concatenate([np.asarray(obj_pcd.normals) for obj_pcd in obj_pcds], axis=0)]
            else:
                return [np.concatenate([np.asarray(obj_pcd.points) for obj_pcd in obj_pcds], axis=0)]

        if return_normals:
            for obj_pcd in obj_pcds:
                obj_pcd.estimate_normals()
            return [np.asarray(obj_pcd.points) for obj_pcd in obj_pcds], [np.asarray(obj_pcd.normals) for obj_pcd in obj_pcds]
        else:
            return [np.asarray(obj_pcd.points) for obj_pcd in obj_pcds]
        
    def T_postprocess(self, obj_pcd, debug=False):
        # discard z axis
        z_mean = np.mean(np.asarray(obj_pcd.points)[:, 2])
        obj_points = np.asarray(obj_pcd.points)
        obj_points[:, 2] = z_mean
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)

        cad_path = 'perception/T-mesh-yz-flat.obj'
        keypoints = np.array([
            [0, 0, 0],
            [0.06, 0, 0],
            [-0.06, 0, 0],
            [0, 0.12, 0],
        ])
        keypoints[:, 2] = z_mean
        keypoint_pcd = o3d.geometry.PointCloud()
        keypoint_pcd.points = o3d.utility.Vector3dVector(keypoints)
        mesh_pcd = o3d.io.read_triangle_mesh(cad_path)
        mesh_pcd.compute_vertex_normals()
        mesh_pcd = mesh_pcd.sample_points_uniformly(number_of_points=10000)
        mesh_pcd = mesh_pcd.voxel_down_sample(voxel_size=0.001)
        mesh_points = np.asarray(mesh_pcd.points)
        mesh_points[:, 2] = z_mean

        # scale T
        mesh_points[:, :2] = mesh_points[:, :2] * T_SCALE
        keypoints[:, :2] = keypoints[:, :2] * T_SCALE
        keypoint_pcd.points = o3d.utility.Vector3dVector(keypoints)

        mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)
        rotation_matrices = [
            o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi * 2 / 180 * i)).astype(np.float32)
            for i in range(180)
        ]

        mesh_center = mesh_pcd.get_center()
        obj_center = obj_pcd.get_center()
        mesh2world = np.eye(4, dtype=np.float32)
        mesh2world[:3, 3] = obj_center - mesh_center
        center_mesh = deepcopy(mesh_pcd)
        center_mesh.transform(mesh2world)

        min_dist = np.inf
        best_result = None
        for rotation_matrix in rotation_matrices:
            copy_mesh = deepcopy(center_mesh)
            rot_center = copy_mesh.get_center()
            local_transformation = np.eye(4, dtype=np.float32)
            local_transformation[:3, :3] = rotation_matrix
            local_transformation[:3, 3] = rot_center - np.dot(rotation_matrix, rot_center)
            copy_mesh.transform(local_transformation)
            dists = copy_mesh.compute_point_cloud_distance(obj_pcd)
            if np.mean(dists) < min_dist:
                min_dist = np.mean(dists)
                best_result = local_transformation
        mesh2world = np.dot(best_result, mesh2world)
        center_mesh.transform(best_result)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            center_mesh,
            obj_pcd,
            0.005,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        mesh2world = np.dot(reg_p2p.transformation, mesh2world)
        center_mesh.transform(reg_p2p.transformation)
        keypoint_pcd.transform(mesh2world)

        if debug:
            coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=obj_center)
            colors = np.array([[1, 0, 0] for _ in range(len(keypoint_pcd.points))])
            keypoint_pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([obj_pcd, keypoint_pcd, center_mesh, coordinates])

        return keypoint_pcd

    def get_all_masks(self, obs, debug=False):
        masks = self.predictor.mask_generation(obs, debug=debug)
        return masks

    def reset_to_default_pose(self):
        action = np.array([-0.143, 0.394, 0.140, -0.397, 0.917, 0.019, -0.007, 1])
        if self.use_robot:
            ## new default, higher position than before
            msg = str(action.tolist()) + "\n"
            self.action_sock.sendall(msg.encode())
            ack = self.action_sock.recv(1024).decode('utf-8').strip()
            if ack != "ACK":
                print(f"⚠️ Reset NACK: {ack}")
        
        print("Reset to default pose:", action)
        # self.robot.set_servo_angle(angle=self.robot_default_pos, speed=20, wait=True)

    def move_to_pose(self, pose, speed=1):
        # speed = speed * 100
        # speed = 60 # TODO: Change this in final results
        # rpy = quat2euler(pose[-4:])
        # offset = self._apply_gripper2hand_offset(pose[:3], rpy)
        # print("Our final control: ", offset)
        # self.robot.set_position(offset[0]*1000, offset[1]*1000, offset[2]*1000, rpy[0], rpy[1], rpy[2], speed=speed, wait=True, is_radian=True)
        if self.use_robot:
            print("Move to pose:", pose)
            msg = str(pose.tolist()) + "\n"
            self.action_sock.sendall(msg.encode())
            ack = self.action_sock.recv(1024).decode().strip()
            if ack != "ACK":
                print(f"⚠️ Move NACK: {ack}")
        return 0

    def move_to_table_position(self, x, y, z, yaw=None, speed=1, wait=True):
        # speed = speed * 100
        # speed = 50 # TODO: Change this in final results
        # if yaw:
        #     rpy = np.array([np.pi, 0, yaw])
        # else:
        #     rpy = np.array([np.pi, 0, 0])

        # position = np.array([x, y, z + Z_PUSHER])
        # offset = self._apply_gripper2hand_offset(position, rpy)
        
        # self.robot.set_position(offset[0]*1000, offset[1]*1000, offset[2]*1000, rpy[0], rpy[1], rpy[2], speed=speed, wait=wait, is_radian=True)
        # time.sleep(4.0)
        print(f"Move to table position: x={x}, y={y}, z={z}, yaw={yaw}, speed={speed}")
        
        if self.use_robot:
            if yaw is not None:
                rpy = np.array([np.pi, 0, yaw])
            else:
                rpy = np.array([np.pi, 0, 0])
            position = np.array([x, y, z])
            # offset = self._apply_gripper2hand_offset(position, rpy)
            msg = str(position.tolist() + rpy.tolist() + [speed]) + "\n"
            self.action_sock.sendall(msg.encode())
            time.sleep(4.0)
            ack = self.action_sock.recv(1024).decode().strip()
            if ack != "ACK":
                print(f"⚠️ Move to table position NACK: {ack}")
        # self.reset_to_default_pose()
        return 0

    def _apply_gripper2hand_offset(self, pose, rpy):
        hand_xyz = pose + euler2mat(rpy[0], rpy[1], rpy[2]) @ np.array([0, 0, -EE_LENGTH])
        return hand_xyz

    def _apply_hand2gripper_offset(self, pose, rpy):
        hand_xyz = pose - euler2mat(rpy[0], rpy[1], rpy[2]) @ np.array([0, 0, -EE_LENGTH])
        return hand_xyz

    def world_to_viewport(self, points, camera_index=0):
        camera_intrinsic = self.get_intrinsic(camera_index)
        points = np.array(points)

        # world to view
        world_to_camera_transforms = np.linalg.inv(self.camera_to_robot_transforms[camera_index])
        vtx_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        view_points = np.dot(vtx_homogeneous, world_to_camera_transforms.T)

        # view to image
        view_points = view_points[:, :3]
        image_points = np.dot(view_points, camera_intrinsic.T)
        image_points = image_points[:, :2] / image_points[:, 2][:, None]
        return image_points

    def ground_position(self, i, depth, x, y):
        depth = depth / 1000
        "ground image position to world position on the table"
        camera_intrinsic = self.get_intrinsic(i)
        z_cam = depth[int(y), int(x)]
        image_point = np.array([x, y, 1]) * z_cam
        cam_point = image_point @ np.linalg.inv(camera_intrinsic).T
        vtx_homo = np.concatenate([cam_point, [1]], axis=0)
        vtx_transformed_homo = np.dot(vtx_homo, self.camera_to_robot_transforms[i].T)
        return vtx_transformed_homo[:3]
