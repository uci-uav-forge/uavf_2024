from uavf_2024.imaging import Camera
import cv2 as cv
import os

cam = Camera()

cam.request_center()
cam.request_autofocus()

os.makedirs("snapshots", exist_ok = True)

for i in range(20):
    img = cam.take_picture()
    cv.imwrite(f"snapshots/{i}.png", img.get_array())
