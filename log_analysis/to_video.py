import cv2 as cv
import traceback
from tqdm import tqdm
from pathlib import Path
from bisect import bisect_left
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Unused, but necessary for 3D plotting
import numpy as np

logs_dir = Path('/home/ericp/uavf_2024/flight_logs/perception_logs_607/06-07 10:39')
video = cv.VideoWriter('video.mp4', cv.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))

pose_stamps = sorted([float(fname.stem) for fname in (logs_dir / 'poses').glob('*.json')])

def draw_orientation(pose):
    orientation = R.from_quat(pose['rotation'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # draw but dont label ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    position = [0,0,0]
    ax.quiver(*position, orientation.apply([1,0,0])[0], orientation.apply([1,0,0])[1], orientation.apply([1,0,0])[2], color='r')
    ax.quiver(*position, orientation.apply([0,1,0])[0], orientation.apply([0,1,0])[1], orientation.apply([0,1,0])[2], color='g')
    ax.quiver(*position, orientation.apply([0,0,1])[0], orientation.apply([0,0,1])[1], orientation.apply([0,0,1])[2], color='b')

    # return fig as image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    plt.close(fig)
    return image

plot_mask = None

def closest_pose(timestamp):
    closest_idx = bisect_left(pose_stamps, timestamp)
    if closest_idx == len(pose_stamps):
        closest_idx -= 1
    pose_stamp = pose_stamps[closest_idx] 
    if abs(pose_stamp - timestamp) > 0.5:
        return None
    pose = json.load(open(logs_dir / 'poses' / f"{pose_stamp:.2f}.json"))
    return pose

for frame_fname in tqdm(sorted((logs_dir / 'camera').glob('*.jpg'))):
    try:

        frame = cv.imread(str(frame_fname))
        timestamp = float(frame_fname.stem)
        pose = closest_pose(timestamp)
        time_mod_10000 = timestamp - (timestamp // 10000) * 10000
        cv.putText(frame, f"Time % 10,000: {time_mod_10000:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        if pose is None:
            cv.putText(frame, "No pose", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        else:
            position_str = f"Position: {[round(x, 2) for x in pose['position']]}"
            cv.putText(frame, position_str, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            orientation_drawing = draw_orientation(pose)
            # draw on top right
            if plot_mask is None:
                # create mask that only includes pixels that aren't pure white in the orientation drawing
                # since it never changes, we just calculate it on the first frame and reuse it
                plot_mask = np.any(orientation_drawing != 255, axis=2)
            frame[0:orientation_drawing.shape[0], -orientation_drawing.shape[1]:][plot_mask] = orientation_drawing[plot_mask]
        video.write(frame)
    except:
        print(f"Error processing frame {frame_fname}")
        traceback.print_exc()

video.release()