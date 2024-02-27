import cv2 as cv
from uavf_2024.imaging import Camera

if __name__ == "__main__":
    cam = Camera()

    while 1:
        img = cam.take_picture()
        attitude = cam.getAttitude()
        cv.putText(img.get_array(), f"Yaw: {attitude[0]} Pitch: {attitude[1]}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("img", img.get_array())
        k = cv.waitKey(30) & 0xff
        if k == 27 or k == ord('q'):
            break
        elif k == ord('a'):
            cam.requestGimbalSpeed(-20,0)
        elif k == ord('d'):
            cam.requestGimbalSpeed(20,0)
        elif k == ord('w'):
            cam.requestGimbalSpeed(0,20)
        elif k == ord('s'):
            cam.requestGimbalSpeed(0,-20)
        else:
            cam.requestGimbalSpeed(0,0)
    
    cam.disconnect()