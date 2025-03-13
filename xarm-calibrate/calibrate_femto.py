import os
import cv2
import numpy as np
import pickle
from typing import Union, Any, Optional

from pyorbbecsdk import Config, Context, OBSensorType, OBFormat, Pipeline, FrameSet, VideoFrame, OBLogLevel

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

class FemtoEnv:
    def __init__(self, vis_dir='calibration_result_femto'):
        self.vis_dir = vis_dir
        self.calibrate_result_dir = vis_dir
        calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        calibration_parameters =  cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(calibration_dictionary, calibration_parameters)
        self.calibration_board = cv2.aruco.GridBoard(
            (5, 7),
            markerLength=0.03439,
            markerSeparation=0.00382,
            dictionary=calibration_dictionary,
        )
        self.setup_femto()

    def setup_femto(self):
        config = Config()
        context = Context()
        context.set_logger_level(OBLogLevel.NONE)
        self.pipeline = Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(3840, 2160, OBFormat.RGB, 15) 
        config.enable_stream(color_profile)
        self.pipeline.start(config)

    def get_obs(self):
        frames = None
        while frames is None:
            frames: FrameSet = self.pipeline.wait_for_frames(100)
        color_frame = frames.get_color_frame()
        color_image = frame_to_bgr_image(color_frame)
        return {'color': color_image}
    
    def get_intrinsics(self):
        camera_param = self.pipeline.get_camera_param()
        intr = camera_param.rgb_intrinsic
        mat = np.eye(3)
        mat[0, 0] = intr.fx
        mat[1, 1] = intr.fy
        mat[0, 2] = intr.cx
        mat[1, 2] = intr.cy
        return mat

    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        if visualize:
            os.makedirs(f'{self.vis_dir}', exist_ok=True)
        
        # Calculate the markers
        obs = self.get_obs()
        intrs = self.get_intrinsics()
        dist_coef = np.zeros(5)

        intr = self.get_intrinsics()
        calibration_img = obs[f'color'].copy()
        if visualize:
            cv2.imwrite(f'{self.vis_dir}/calibration_img.jpg', calibration_img)
        
        calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
        detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
            detectedCorners=corners, 
            detectedIds=ids,
            rejectedCorners=rejected_img_points,
            image=calibration_img,
            board=self.calibration_board,
            cameraMatrix=intr,
            distCoeffs=dist_coef,
        )

        if visualize:
            calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
            cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_femto.jpg', calibration_img_vis)

        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners=detected_corners,
            ids=detected_ids,
            board=self.calibration_board,
            cameraMatrix=intr,
            distCoeffs=dist_coef,
            rvec=None,
            tvec=None,
        )

        if not retval:
            print("pose estimation failed")
            import pdb; pdb.set_trace()

        if visualize:
            calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
            cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef, rvec, tvec, 0.1)
            cv2.imwrite(f"{self.vis_dir}/calibration_result.jpg", calibration_img_vis)

        if save:
            # save rvecs, tvecs
            with open(f'{self.calibrate_result_dir}/rvec.pkl', 'wb') as f:
                pickle.dump(rvec, f)
            with open(f'{self.calibrate_result_dir}/tvec.pkl', 'wb') as f:
                pickle.dump(tvec, f)
        if return_results:
            return rvec, tvec
        
        self.pipeline.stop()
        
femto_env = FemtoEnv()
femto_env.fixed_camera_calibrate(visualize=True, save=True, return_results=False)
