import pygame
import numpy as np
import cv2 as cv
from time import strftime, time
from uavf_2024.imaging import Camera
import os

cam = Camera()
cam.setAbsoluteZoom(1)
is_recording = False
target_zoom_level = cam.getZoomLevel()
speed_multiplier = 20

imgs_dir = f"recorded_images/{strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(imgs_dir, exist_ok=True)

pygame.init()
pygame.joystick.init()
X = 1600
Y = 900
 
# create the display surface object
# of specific dimension..e(X, Y).
screen = pygame.display.set_mode((X, Y), pygame.DOUBLEBUF)
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
        slew_x = clamped*speed_multiplier
    
    if LEFT_AXIS_Y in axis and abs(axis[LEFT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_Y]) # use tanh to clamp to [-1,1]
        slew_y = clamped*speed_multiplier

    # duplicate for right axis
    if RIGHT_AXIS_X in axis and abs(axis[RIGHT_AXIS_X])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x += clamped
    
    if RIGHT_AXIS_Y in axis and abs(axis[RIGHT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_Y])
        slew_y += clamped

    cam.requestGimbalSpeed(int(slew_x), -int(slew_y))

def handle_button_press(button: int):
    global is_recording, target_zoom_level, speed_multiplier
    if button == BUTTON_CIRCLE:
        is_recording = not is_recording
    
    if button == BUTTON_RIGHT_TRIGGER:
        if target_zoom_level < 10:
            target_zoom_level += 1
            cam.setAbsoluteZoom(target_zoom_level)
    
    if button == BUTTON_RIGHT_BUMPER:
        if target_zoom_level > 1:
            target_zoom_level -= 1
            cam.setAbsoluteZoom(target_zoom_level)

    if button == BUTTON_LEFT_TRIGGER:
        if speed_multiplier < 100:
            speed_multiplier += 10
        
    if button == BUTTON_LEFT_BUMPER:
        if speed_multiplier > 10:
            speed_multiplier -= 10

def main():
    global is_recording
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
        if event.type == pygame.JOYBUTTONDOWN:
            handle_button_press(event.button) 
        # if press q, quit
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return False


    img = cam.get_latest_image().get_array()
    yaw, pitch, _roll = cam.getAttitude()
    
    if is_recording:
        img_num = len(os.listdir(imgs_dir))
        cv.imwrite(f"{imgs_dir}/{img_num}.jpg", img)
        with open(f"{imgs_dir}/{img_num}.txt", "w") as f:
            f.write("\n".join([
                f"{yaw:.4f},{pitch:.4f}",
                str(cam.getFocalLength()).
                str(time())
            ]))

    joystick_control()

    # # draw current position
    cv.rectangle(img, (10,50), (200,0), (0,0,0), -1)
    readout_text = f"\nyaw: {yaw:.2f} pitch: {pitch:.2f}\n zoom: {target_zoom_level} speed: {speed_multiplier}\n"
    for i, line in enumerate(readout_text.split('\n')):
        cv.putText(img, line, (10,20+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    if X != img.shape[1] or Y != img.shape[0]:
        img = cv.resize(img, (X,Y))

    # # draw a crosshair
    h,w = img.shape[:2]
    cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
    cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)

    circle_thickness = -1 if is_recording else 2
    cv.circle(img, (w-30,30),20,(0,0,255),circle_thickness)

    pygame.display.flip()
    # display image in pygame window
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.rot90(img)
    img = np.flipud(img)
    img = pygame.surfarray.make_surface(img)
    screen.blit(img, (0,0))
    
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