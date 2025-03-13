import pickle
import numpy as np
import cv2

rvecs = pickle.load(open('real_world/calibration_result/rvecs.pkl', 'rb'))
tvecs = pickle.load(open('real_world/calibration_result/tvecs.pkl', 'rb'))
hand_eye = pickle.load(open('real_world/calibration_result/calibration_handeye_result.pkl', 'rb'))
serial_numbers = list(rvecs.keys())

camera_to_bases = {}

for serial_number in serial_numbers:
    rvec = rvecs[serial_number]
    tvec = np.squeeze(tvecs[serial_number])
    R_mtx, _ = cv2.Rodrigues(rvec)
    world_to_camera = np.array([[R_mtx[0][0], R_mtx[0][1], R_mtx[0][2], tvec[0]],
                      [R_mtx[1][0], R_mtx[1][1], R_mtx[1][2], tvec[1]],
                      [R_mtx[2][0], R_mtx[2][1], R_mtx[2][2], tvec[2]],
                      [0, 0, 0, 1]])
    camera_to_world = np.linalg.inv(world_to_camera)
    base_to_world = np.concatenate((hand_eye['R_base2world'], np.expand_dims(hand_eye['t_base2world'], axis=1)), axis=1)
    base_to_world = np.concatenate((base_to_world, np.array([[0, 0, 0, 1]])), axis=0)
    camera_to_base = np.dot(np.linalg.inv(base_to_world), camera_to_world)
    camera_to_bases[serial_number] = camera_to_base

with open('real_world/calibration_result/camera_to_bases.pkl', 'wb') as f:
    pickle.dump(camera_to_bases, f)

print("Camera to base transforms saved to real_world/calibration_result/camera_to_bases.pkl")