import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import pickle
import random

from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Value, Manager
from threading import Lock
from camera.multi_realsense import MultiRealsense, SingleRealsense
from camera.video_recorder import VideoRecorder
from xarm6 import XARM6
import sapien.core as sapien

from common.kinematics_utils import KinHelper
from pynput import keyboard, mouse

# Global variable to track key states
key_states = {
    "w": False,
    "s": False,
    "a": False,
    "d": False,
    "up": False,    # Up arrow key for moving up
    "down": False,  # Down arrow key for moving down
    "z": False,     # Key for adjusting roll
    "x": False,     # Key for adjusting pitch
    "c": False,      # Key for adjusting yaw
    "q": False,     # Key for adjusting yaw
    "e": False      # Key for adjusting yaw
}



# Function to handle keyboard events
def on_press(key):
    try:
        key_char = key.char.lower() if key.char else key.char
        if key_char in key_states:
            key_states[key_char] = True
    except AttributeError:
        if key == keyboard.Key.up:
            key_states["up"] = True
        elif key == keyboard.Key.down:
            key_states["down"] = True

def on_release(key):
    try:
        key_char = key.char.lower() if key.char else key.char
        if key_char in key_states:
            key_states[key_char] = False
    except AttributeError:
        if key == keyboard.Key.up:
            key_states["up"] = False
        elif key == keyboard.Key.down:
            key_states["down"] = False
        if key == keyboard.Key.esc:
            return False



def listen_keyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()



# Initialize and start the listener threads
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()


class ImageWriter(mp.Process):

    def __init__(self, index, realsense, capture_fps, record_fps, record_time, record_flag):
        super().__init__()
        self.index = index
        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time

        self.realsense = realsense
        self.record_flag = record_flag
        # self.robot_obs = robot_obs

        self.lock = Lock()

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        realsense = self.realsense
        # robot_obs = self.robot_obs

        out = None
        next_step_idx = 0

        f = open(f'recording/timestamps_{i}.txt', 'a')

        # action = open(f'recording/actions.txt', 'a')

        while self.alive:
            try:
                out = realsense.get(out=out)
                if out[i]['step_idx'] == next_step_idx * (capture_fps // record_fps):
                    print(f'step {out[i]["step_idx"]} camera {i} step {out[i]["step_idx"]} timestamp {out[i]["timestamp"]}')
                    timestamp = out[i]['timestamp']

                    with self.lock:
                        cv2.imwrite(f'recording/camera_{i}/{next_step_idx:06}.jpg', out[i]['color'])
                        cv2.imwrite(f'recording/camera_{i}/{next_step_idx:06}_depth.png', out[i]['depth'])
                        f.write(f'{timestamp}\n')
                        f.flush()

                    next_step_idx += 1
                    if i == 0:
                        self.record_flag.value = True

                if next_step_idx >= record_time * record_fps:
                    f.close()
                    self.alive = False
                    # self.record_flag.value = False

            except Exception as e:
                print(f"Error in camera {i}: {e}")
                f.close()
                self.alive = False
                self.record_flag.value = False
            

    def start(self):
        self.alive = True
        super().start()

    def stop(self):
        self.alive = False
    
    def join(self):
        super().join()

def random_point_in_rectangle(top_left, top_right, bottom_left, bottom_right):
    # Assuming the rectangle is axis-aligned, we find the min and max for x and y coordinates
    min_x = min(top_left[0], bottom_left[0])
    max_x = max(top_right[0], bottom_right[0])
    min_y = min(top_left[1], top_right[1])
    max_y = max(bottom_left[1], bottom_right[1])

    # Randomly sample a point within the rectangle
    random_x = random.uniform(min_x, max_x)
    random_y = random.uniform(min_y, max_y)

    return random_x, random_y

def generate_action_sequence(num_actions, Z_up, Z_down, top_left, top_right, bottom_left, bottom_right):
    action_sequence = []
    min_x = min(top_left[0], bottom_left[0])
    max_x = max(top_right[0], bottom_right[0])
    min_y = min(top_left[1], top_right[1])
    max_y = max(bottom_left[1], bottom_right[1])
    
    # Calculate the center of the rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    for _ in range(num_actions):
        # Selecting start position (x1, y1) randomly in one of the four subrectangles
        if random.choice([True, False]):
            x1 = random.uniform(min_x, center_x)  # Left half
        else:
            x1 = random.uniform(center_x, max_x)  # Right half

        if random.choice([True, False]):
            y1 = random.uniform(min_y, center_y)  # Bottom half
        else:
            y1 = random.uniform(center_y, max_y)  # Top half

        # Ensure end position (x2, y2) is in the diagonally opposite subrectangle
        if x1 < center_x:
            x2 = random.uniform(center_x, max_x)
        else:
            x2 = random.uniform(min_x, center_x)

        if y1 < center_y:
            y2 = random.uniform(center_y, max_y)
        else:
            y2 = random.uniform(min_y, center_y)

        # Define the movements for one complete action
        move_to_Z_up = [x1, y1, Z_up, 179.3, 0, -0.1]
        move_to_Z_down = [x1, y1, Z_down, 179.3, 0, -0.1]
        move_back_to_Z_down = [x2, y2, Z_down, 179.3, 0, -0.1]
        move_to_next_Z_up = [x2, y2, Z_up, 179.3, 0, -0.1]

        # Add the movements to the action sequence
        action_sequence.extend([move_to_Z_up, move_to_Z_down, move_back_to_Z_down, move_to_next_Z_up])

    return action_sequence

def test():

    # WH = [640, 480]
    WH = [1280, 720]

    capture_fps = 15
    record_fps = 15

    assert capture_fps % record_fps == 0
    
    WRIST = '246322303938'  # device id of the wrist camera (connected to the PC but we want to ignore it)
    
    serial_numbers = SingleRealsense.get_connected_devices_serial()
    # serial_numbers.remove(WRIST)
    
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    realsense =  MultiRealsense(
            serial_numbers=serial_numbers,
            shm_manager=shm_manager,
            resolution=(WH[0], WH[1]),
            capture_fps=capture_fps,
            record_fps=record_fps,
            enable_color=True,
            enable_depth=True,
            # get_max_k=1,  # only the latest frame is needed
            # advanced_mode_config=config,
            # transform=transform,
            # recording_transform=transform,
            # video_recorder=video_recorder,
            verbose=False)

    try:
        realsense.start()
        realsense.set_exposure(exposure=100, gain=60)

        
        exposure_time = 5
        rec_start_time = time.time() + exposure_time
        realsense.restart_put(start_time=rec_start_time)

        record_time = 100

        for i in range(len(serial_numbers)):
            os.makedirs(f'recording/camera_{i}', exist_ok=True)
            if os.path.exists(f'recording/timestamps_{i}.txt'):
                os.remove(f'recording/timestamps_{i}.txt')

        print('start recording')

        manager = Manager()

        # gripper_enable = False
        # speed = 100
        # robot = XARM6(speed = speed, gripper_enable=gripper_enable)
        ip = "192.168.1.209"
        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(1)
        arm.set_state(state=0)
        kin_helper = KinHelper(robot_name='xarm6')

        time.sleep(1)

        # x, y, z, roll, pitch, yaw = [463.3,-1.6,434,179.2,0,0.3]
        x, y, z, roll, pitch, yaw = [533.9,96.2,434,179.2,0,0.3]
        # x, y, z, roll, pitch, yaw = [196.2,-1.6,434,179.2,0,0.3]
        # x, y, z, roll, pitch, yaw = [396.3,158.6,322.1,177.7,-0.3,0.7]
        initial_qpos = arm.get_servo_angle()[1][0:6]
        next_position = [x, y, z, roll, pitch, yaw]
        initial_qpos = np.array(initial_qpos) / 180. * np.pi
        next_position_unit = np.zeros_like(np.array(next_position))
        next_position_unit[0:3] = np.array(next_position)[0:3] / 1000.
        next_position_unit[3:] = np.array(next_position)[3:] / 180. * np.pi
        next_servo_angle = kin_helper.compute_ik_sapien(initial_qpos, next_position_unit)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx",next_servo_angle)
        for i in range(1000):
            angle = initial_qpos + (next_servo_angle - initial_qpos) * i / 1000.
            # print(angle)
            # print(arm.get_position())
            arm.set_servo_angle_j(angles=angle,is_radian=True)
            time.sleep(0.001)
        
        use_robot = True
        record_flag = manager.Value(ctypes.c_bool, False)
        robot_action_flag = manager.Value(ctypes.c_bool, True)

        writers = []
        for i in range(len(serial_numbers)):
            writer = ImageWriter(
                index=i,
                realsense=realsense,
                capture_fps=capture_fps,
                record_fps=record_fps,
                record_time=record_time,
                record_flag=record_flag,
            )
            writer.start()
            writers.append(writer)
            # write the serial number to each camera folder
            with open(f'recording/camera_{i}/serial_number.txt', 'w') as f:
                f.write(serial_numbers[i])


        if os.path.exists(f'recording/actions.txt'):
            os.remove(f'recording/actions.txt')



        save_action = open(f'recording/actions.txt', 'a')

        init_pose = arm.get_position()
        x, y, z, roll, pitch, yaw = init_pose[1]

        angle_step = 0.1
        rz_angle = 0
        running = True
        while running:
            if key_states["w"]:
                x += 1
            if key_states["s"]:
                x -= 1
            if key_states["a"]:
                y += 1
            if key_states["d"]:
                y -= 1
            if key_states["up"]:
                z += 1   # Move up
            if key_states["down"]:
                z -= 1   # Move down
            if key_states["z"]:
                roll += angle_step
            if key_states["x"]:
                pitch += angle_step
            if key_states["c"]:
                yaw += angle_step
            if key_states["q"]:
                rz_angle += angle_step * 0.1
            if key_states["e"]:
                rz_angle -= angle_step * 0.1

            initial_qpos = arm.get_servo_angle()[1][0:6]
            print(initial_qpos)
            if z < 110: z = 110 # this limit works for the long white stick
            # if z < 50: z = 50 # this limit works for the short white stick
            # if z < 40: z = 40 # this limit works for the black pusher
            next_position = [x, y, z, roll, pitch, yaw]
            initial_qpos = np.array(initial_qpos) / 180. * np.pi
            next_position_unit = np.zeros_like(np.array(next_position))
            next_position_unit[0:3] = np.array(next_position)[0:3] / 1000.
            next_position_unit[3:] = np.array(next_position)[3:] / 180. * np.pi
            next_servo_angle = kin_helper.compute_ik_sapien(initial_qpos, next_position_unit)
            joints = next_servo_angle
            next_servo_angle[-1] += rz_angle

            for i in range(100):
                angle = initial_qpos + (next_servo_angle - initial_qpos) * i / 100.
                arm.set_servo_angle_j(angles=angle,is_radian=True, speed=0.2)
            sleep_time = 0.001
            time.sleep(sleep_time)
            arm.clean_error()
            arm.clean_warn()

            if not keyboard_listener.is_alive():
                break

            if robot_action_flag.value == True:
                if record_flag.value == True:
                    robot_obs = dict()
                    robot_obs['joint_angles'] = arm.get_servo_angle()[1][0:6]
                    robot_obs['pose'] = arm.get_position()[1]
                    save_action.write(f'{json.dumps(robot_obs)}\n')
                    record_flag.value = False

        for writer in writers:
            writer.join()


    finally:
        realsense.stop()
    

    for i in range(len(serial_numbers)):
        os.system('ffmpeg -loglevel panic -r 15 -f image2 -s 1280x720 -pattern_type glob -i "recording/camera_{i}/*.jpg" -vcodec libx264 -crf 25 -pix_fmt yuv420p "recording/camera_{i}/all.mp4" -y')


if __name__ == "__main__":
    test()
