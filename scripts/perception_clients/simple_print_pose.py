import asyncio
from mavsdk import System # pip install mavsdk
from scipy.spatial.transform import Rotation as R
from uavf_2024.imaging import Camera
import os
from time import strftime, time
import cv2 as cv

dirname = f"logs_{strftime('%H_%M_%S')}"

os.makedirs(dirname, exist_ok=True)

async def print_drone_position():
    drone = System()
    cam = Camera()
    cam.request_down()
    # Connect to the drone system
    # if this hangs, it's probably because /dev/ttyACM0 doesn't have the right perms.
    # to fix, do `sudo chmod 777 /dev/ttyACM0`
    await drone.connect(system_address="serial:///dev/ttyACM0:921600")

    angles = None

    # Start receiving telemetry updates
    last_autofocus = time()
    frame_no = 0
    try:
        async for telemetry in drone.telemetry.attitude_quaternion():
            # print(f"Position (Quaternion): w: {telemetry.w:.4f}, x: {telemetry.x:.4f}, y: {telemetry.y:.4f}, z: {telemetry.z:.4f}")
            r = R.from_quat([telemetry.x, telemetry.y, telemetry.z, telemetry.w])
            angles = r.as_euler('zyx', degrees=True)
            async for telemetry in drone.telemetry.position_velocity_ned():
                if time() - last_autofocus > 1:
                    cam.request_down()
                    cam.request_autofocus()
                    last_autofocus = time()
                img = cam.get_latest_image()
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear for Windows/Linux
                msg = f"Drone angles (zyx): {[f'{a:.2f}' for a in angles]}"
                position_ned = telemetry.position
                vel_ned = telemetry.velocity
                pos = [position_ned.north_m, position_ned.east_m, position_ned.down_m]
                vel = [vel_ned.north_m_s, vel_ned.east_m_s, vel_ned.down_m_s]
                msg += f"\nDrone pos (NED): {[f'{p:.2f}' for p in pos]}"
                cam_angles = cam.getAttitude()
                msg+= f"\nCamera angles: {[f'{a:.2f}' for a in cam_angles]}"
                print(msg, end='\r')

                time_ms = int(time()*1000)

                cv.imwrite(f"{dirname}/{frame_no}.png", img.get_array())
                with open(f"{dirname}/{frame_no}.txt", "w+") as f:
                    f.write(f"{time_ms}\n")
                    f.write(f"{','.join(map(str,r.as_quat()))}\n")
                    f.write(f"{','.join(map(str,cam_angles))}\n")
                    f.write(f"{','.join(map(str,pos))}\n")
                    f.write(f"{','.join(map(str,vel))}\n")
                frame_no += 1
    except:
        return



if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(print_drone_position())