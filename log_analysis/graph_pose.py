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

dir_name= Path("/home/ericp/uavf_2024/flight_logs/perception_logs_607/06-07 10:39")


for fname in tqdm(sorted(os.listdir(dir_name / 'poses'))):
    pose = json.load(open(dir_name / 'poses' / fname))
    position = pose['position']
    rotation_quat = pose['rotation']
    rotation = R.from_quat(rotation_quat)
    euler = rotation.as_euler('xyz', degrees=True)
    timestamp = pose['time_seconds']

    pose_timestamps.append(timestamp)
    x_positions.append(position[0])
    y_positions.append(position[1])
    z_positions.append(position[2])

    x_rotations.append(euler[0])
    y_rotations.append(euler[1])
    z_rotations.append(euler[2])

alt_timestamps = []
alt_msl = []
alt_local = []
alt_rel = []
alt_terrain = []

for fname in tqdm(sorted(os.listdir(dir_name / 'altitudes'))):
    alt = json.load(open(dir_name / 'altitudes' / fname))
    timestamp = float(fname[:-5])
    msl = alt['amsl']
    local = alt['local']
    rel = alt['relative']
    terrain = alt['terrain']

    alt_timestamps.append(timestamp)
    alt_msl.append(msl)
    alt_local.append(local)
    alt_rel.append(rel)
    alt_terrain.append(terrain)

for (y_name, y_arr) in zip(["msl", "local", "relative", "terrain"], [alt_msl, alt_local, alt_rel, alt_terrain]):
    plt.figure()
    plt.plot(alt_timestamps, y_arr, label=y_name)
    plt.title(f"{y_name} altitude vs time")
    plt.savefig(f"{y_name}_vs_time.png")

for (y_name, y_arr) in zip(["x_positions", "y_positions", "z_positions", "x_rotations", "y_rotations", "z_rotations"], [x_positions, y_positions, z_positions, x_rotations, y_rotations, z_rotations]):
    plt.figure() 
    plt.plot(pose_timestamps, y_arr, label= y_name)
    plt.title(f"{y_name} vs time")
    plt.savefig(f"{y_name}_vs_time.png")

plt.figure()
plt.title("xy positions")
plt.xlabel("x")
plt.ylabel("y") 
plt.scatter(x_positions, y_positions, c=pose_timestamps, cmap='viridis', s=1)
plt.savefig("xy_positions.png")