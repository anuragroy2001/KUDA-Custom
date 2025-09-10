import os
import glob
import numpy as np
from PIL import Image   # Ensure you have Pillow installed: pip install Pillow  

# Define the directory containing the video frames
video_dir = '/home/panasonic/KUDA/logs/rope-planning-1752313653.2399294/tracking_0'   

frames = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))[::10]
video = []
for frame in frames:
    img = Image.open(frame)
    video.append(np.array(img))
video = np.stack(video, axis=0)

# get every 10th depth frame
depth_files = sorted(glob.glob(os.path.join(video_dir, '*.png')))[::10]
depths = []
for depth_file in depth_files:
    depth = Image.open(depth_file)
    depths.append(np.array(depth) / 1000.)
depths = np.stack(depths, axis=0)