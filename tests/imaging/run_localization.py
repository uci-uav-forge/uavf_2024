import os
from uavf_2024.imaging import Camera, Localizer
from scipy.spatial.transform import Rotation as R
from itertools import product
import numpy as np
import cv2 as cv

CURRENT_FILEPATH = os.path.dirname(os.path.realpath(__file__))
files_dir = f"{CURRENT_FILEPATH}/2024_test_data/arc_test"
visualizations_dir = f"{CURRENT_FILEPATH}/visualizations/localizations"
os.makedirs(visualizations_dir, exist_ok=True)

localizer = Localizer.from_focal_length(Camera.focalLengthFromZoomLevel(1), (1920, 1080))

for i in range(len(os.listdir(files_dir))):
    img_file = f"{files_dir}/img_{i}/bounding_boxes.png"
    pose_file = f"{files_dir}/img_{i}/cam_pose.txt"
    img = cv.imread(img_file)
    with open(pose_file, 'r') as f:
        lines = f.readlines()

        position = tuple(map(float, lines[0].split(',')))
        drone_orientation = tuple(map(float, lines[1].split(',')))
        cam_orientation = tuple(map(float, lines[2].split(',')))
        start_angles = tuple(map(float, lines[3][1:-2].split(', ')))
        end_angles = tuple(map(float, lines[4][1:-2].split(', ')))

        for x,y in product(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21)):
            coords_2d = localizer.coords_to_2d(np.array([x,y,0]), (position, R.from_quat(cam_orientation)))
            coords_2d_int = tuple(map(int, coords_2d))
            if 0<=coords_2d[0]<1920 and 0<=coords_2d[1]<1080:
                cv.putText(img, f"{int(x)},{int(y)}", coords_2d_int, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.circle(img, coords_2d_int, 5, (255,255,0), -1)
        cv.imwrite(f"{visualizations_dir}/img_{i}_localization.png", img) 