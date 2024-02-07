from uavf_2024.imaging import Camera
import cv2 as cv
import os


zoom_level = 1
dirname = f"cam_imgs_zoom_{zoom_level}"
os.makedirs(dirname, exist_ok=True)

cam = Camera()
cam.setAbsoluteZoom(zoom_level)

cv.waitKey(0)

for i in range(10):
    img = cam.take_picture()
    img_cv = img.get_array()
    img_cv = cv.resize(img_cv, (1920//2, 1080//2))

    cv.imshow("Preview", img_cv)
    key = cv.waitKey(0)
    while key == 114: # pressing r to retake
        img = cam.take_picture()
        img_cv = img.get_array()
        img_cv = cv.resize(img_cv, (1920//2, 1080//2))

        cv.imshow("Preview",img_cv)
        key = cv.waitKey(0)
        print("Retaking")
    
    print(f"Saved image {i+1}/10")

    cv.imwrite(f"{dirname}/img{i}.png",img.get_array())

print("Done!")

cam.disconnect()