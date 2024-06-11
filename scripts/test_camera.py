from uavf_2024.imaging import Camera
import time

cam = Camera("test_image_log")
cam.setAbsoluteZoom(1)
cam.request_autofocus()

print("Started recording...")
cam.start_recording()
start_time = time.time()
minutes = 5
while (time.time()-start_time<(minutes*60)):
    time.sleep(0.2)
    print(f"{(time.time()-start_time) / 60}\r\n")
    pass
cam.stop_recording()
