from uavf_2024.imaging import Camera
import time
import cv2 as cv
import os
from tqdm import tqdm

cam = Camera()
cam.start_recording()

cam.setAbsoluteZoom(1)
cam.request_center()
cam.request_autofocus()
print("got past autofocus")
print(cam.getZoomLevel())

os.makedirs("snapshots", exist_ok = True)

for i in tqdm(range(20)):
    img = cam.get_latest_image()
    if img is not None:
        img.save(f"snapshots/img_{i}.jpg")
    else:
        print("image is none")
    time.sleep(0.1)

cam.disconnect()
