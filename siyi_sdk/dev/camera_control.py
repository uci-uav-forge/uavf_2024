import pygame
from siyi_sdk import SIYISDK
import time

class CameraControl:

    def __init__(self, camera : SIYISDK):
        pygame.init()
        if not cam.connect():
            print("No connection ")
            exit(1)
        self.camera = camera
        self._running = True

    def run(self):
        try:
            self.camera.requestLockMode()
            self.clock = pygame.time.Clock()
            self.running = True
            while self.running:
                self.clock.tick(10)
                self.update()
        finally:
            self.camera.disconnect()
            pygame.quit()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            self.running = False
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.requestGimbalSpeed(0,20)
            time.sleep(0.2)
            self.camera.requestGimbalSpeed(0,0)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.requestGimbalSpeed(0,-20)
            time.sleep(0.2)
            self.camera.requestGimbalSpeed(0,0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.requestGimbalSpeed(20,0)
            time.sleep(0.2)
            self.camera.requestGimbalSpeed(0,0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.requestGimbalSpeed(-20,0)
            time.sleep(0.2)
            self.camera.requestGimbalSpeed(0,0)
        if keys[pygame.K_EQUALS]:
            self.camera.requestZoomIn()
            time.sleep(0.2)
            self.camera.requestZoomHold()
            print("Zoom level: ", self.camera.getZoomLevel())
        if keys[pygame.K_MINUS]:
            self.camera.requestZoomOut()
            time.sleep(0.2)
            self.camera.requestZoomHold()
            print("Zoom level: ", self.camera.getZoomLevel())
        if keys[pygame.K_SPACE]:
            self.camera.requestAutoFocus()
            print("Auto focus")
        if keys[pygame.K_c]:
            self.camera.requestCenterGimbal()
            print("Centered gimbal")
        if keys[pygame.K_r]:
            self.camera.requestAbsolutePosition(0, 0)
            print("Pointing gimbal down")

if __name__ == "__main__":
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    camera = CameraControl(cam)
    camera.run()