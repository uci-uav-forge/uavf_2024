from uavf_2024.imaging import Camera
import time
from pathlib import Path

base_path = Path('/home/forge/ws/logs/')
#base_path = Path('/media/forge/SANDISK/logs/')
logs_path = Path(base_path / f'{time.strftime("%m-%d %Hh%Mm")}')
cam = Camera(logs_path / "camera")
cam.setAbsoluteZoom(2)
cam.request_autofocus()

cam.start_recording()
start_time = time.time()
minutes = 5
while(time.time()-start_time < minutes*60):
    print((time.time()-start_time) / 60, end='\r')
    cam.request_autofocus()
    time.sleep(1)
cam.stop_recording()