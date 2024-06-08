import os
import json
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2 as cv
import numpy as np

pose_timestamps = []
x_positions = []
y_positions = []
z_positions = []

x_rotations = []
y_rotations = []
z_rotations = []

img_timestamps = []
img_pixel_diff = []

dir_name= Path("arc_1")
for frame_fname in tqdm(sorted(os.listdir(dir_name / 'camera'))):
    timestamp = float(frame_fname[:-4]) 
    img_timestamps.append(timestamp)
    img = cv.imread(str(dir_name / 'camera' / frame_fname))

img_file_ls = sorted(os.listdir(dir_name / 'camera'))
for iter in tqdm(range(len(img_file_ls))):
    if iter == 0:
        img_pixel_diff.append(0)
    else:
        img_1 = cv.imread(str(dir_name / 'camera' / img_file_ls[iter])).astype(np.float64)
        img_2 = cv.imread(str(dir_name / 'camera' / img_file_ls[iter-1])).astype(np.float64)
        pixel_diff = np.sum(np.power((img_1 - img_2), 2))
        img_pixel_diff.append(pixel_diff)

plt.figure()
plt.plot(img_timestamps, img_pixel_diff, label="Pixel Difference")
plt.title('Pixel Difference vs Timestamps')
plt.legend(['pixel difference', 'phase difference'])


for fname in tqdm(sorted(os.listdir(dir_name / 'poses'))[-250:-150]):
    pose = json.load(open(dir_name / 'poses' / fname))
    position = pose['position']
    rotation_quat = pose['rotation']
    rotation = R.from_quat(rotation_quat)
    euler = rotation.as_euler('xyz', degrees=True)
    timestamp = pose['time_seconds']
    print(timestamp)

    pose_timestamps.append(timestamp)
    x_positions.append(position[0])
    y_positions.append(position[1])
    z_positions.append(position[2])

    x_rotations.append(euler[0])
    y_rotations.append(euler[1])
    z_rotations.append(euler[2])

for (y_name, y_arr) in zip(["x_positions", "y_positions", "z_positions", "x_rotations", "y_rotations", "z_rotations"], [x_positions, y_positions, z_positions, x_rotations, y_rotations, z_rotations]):
    
    plt.plot(pose_timestamps, y_arr, label= y_name)
    plt.savefig(f"{y_name}_vs_time.png")