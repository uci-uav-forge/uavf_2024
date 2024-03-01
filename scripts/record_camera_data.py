import pygame
import numpy as np
import cv2 as cv
from time import strftime
from uavf_2024.imaging import Camera
import os

cam = Camera()
is_recording = False

imgs_dir = "recorded_images"
os.makedirs("", exist_ok=True)

pygame.init()
pygame.joystick.init()
print(f"Joysticks: {pygame.joystick.get_count()}")
controller = pygame.joystick.Joystick(0)
controller.init()

axis = {}

LEFT_AXIS_X = 0
LEFT_AXIS_Y = 1
RIGHT_AXIS_X = 3
RIGHT_AXIS_Y = 4

BUTTON_X = 0
BUTTON_CIRCLE = 1
BUTTON_TRIANGLE = 2
BUTTON_SQUARE = 3

BUTTON_LEFT_BUMPER = 4
BUTTON_RIGHT_BUMPER = 5

BUTTON_LEFT_TRIGGER = 6
BUTTON_RIGHT_TRIGGER = 7

def joystick_control():
    slew_x = 0
    slew_y = 0
    
    if LEFT_AXIS_X in axis and abs(axis[LEFT_AXIS_X])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x = clamped*8
    
    if LEFT_AXIS_Y in axis and abs(axis[LEFT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_Y]) # use tanh to clamp to [-1,1]
        slew_y = clamped*6

    # duplicate for right axis
    if RIGHT_AXIS_X in axis and abs(axis[RIGHT_AXIS_X])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x += clamped
    
    if RIGHT_AXIS_Y in axis and abs(axis[RIGHT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_Y])
        slew_y += clamped

    cam.requestGimbalSpeed(slew_x, slew_y)

def handle_button_press(button: int):
    global is_recording
    if button == BUTTON_CIRCLE:
        is_recording = not is_recording

def main():
    global is_recording
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
        if event.type == pygame.JOYBUTTONDOWN:
            handle_button_press(event.button) 


    img = cam.take_picture()
    
    joystick_control()

    # # draw current position
    cv.rectangle(img, (10,50), (200,0), (0,0,0), -1)
    yaw, pitch, _roll = cam.getAttitude()
    readout_text = ""
    readout_text += f"\nyaw: {yaw:.2f} pitch: {pitch:.2f}"
    for i, line in enumerate(readout_text.split('\n')):
        cv.putText(img, line, (10,20+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if is_recording:
        img_num = len(os.listdir(imgs_dir))
        cv.imwrite(f"{imgs_dir}/{img_num}.jpg", img)
        with open(f"{imgs_dir}/{img_num}.txt", "w") as f:
            f.writelines([
                f"{yaw:.4f},{pitch:.4f}",
                strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ])

    # resize to half size
    scale_factor = 1
    img = cv.resize(img, (1920//scale_factor, 1080//scale_factor))

    # # draw a crosshair
    h,w = img.shape[:2]
    cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
    cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)

    circle_thickness = -1 if is_recording else 2
    cv.circle(img, (w-30,30),20,(0,0,255),circle_thickness)

    cv.imshow("Camera Image", img)

    # if press q, break

    if cv.waitKey(1) == ord('q'):
        return False
    
    return True


# main loop
while True:
    try:
        if not main():
            break
    except Exception as e:
        print(e)
        break

cam.disconnect()