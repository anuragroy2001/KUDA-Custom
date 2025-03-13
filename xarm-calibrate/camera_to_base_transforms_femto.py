import pickle
import numpy as np
import cv2

rvec = pickle.load(open('calibration_result_femto/rvec.pkl', 'rb'))
tvec = pickle.load(open('calibration_result_femto/tvec.pkl', 'rb'))
hand_eye = pickle.load(open('real_world/calibration_result/calibration_handeye_result.pkl', 'rb'))

camera_to_bases = {}

tvec = np.squeeze(tvec)
R_mtx, _ = cv2.Rodrigues(rvec)
world_to_camera = np.array([[R_mtx[0][0], R_mtx[0][1], R_mtx[0][2], tvec[0]],
                    [R_mtx[1][0], R_mtx[1][1], R_mtx[1][2], tvec[1]],
                    [R_mtx[2][0], R_mtx[2][1], R_mtx[2][2], tvec[2]],
                    [0, 0, 0, 1]])
camera_to_world = np.linalg.inv(world_to_camera)
base_to_world = np.concatenate((hand_eye['R_base2world'], np.expand_dims(hand_eye['t_base2world'], axis=1)), axis=1)
base_to_world = np.concatenate((base_to_world, np.array([[0, 0, 0, 1]])), axis=0)
camera_to_base = np.dot(np.linalg.inv(base_to_world), camera_to_world)
camera_to_bases['femto'] = camera_to_base

with open('calibration_result_femto/camera_to_bases.pkl', 'wb') as f:
    pickle.dump(camera_to_bases, f)

print("Camera to base transforms saved to calibration_result_femto/camera_to_bases.pkl")
