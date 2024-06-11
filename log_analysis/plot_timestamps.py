import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import json

def get_images(images_dir: Path):
    timestamps: list[float] = []
    images: list[np.ndarray] = []

    first = None
    
    for path in tqdm(sorted(images_dir.glob("*.json"))):
        timestamp = float(path.stem)
        if first is None:
            first = timestamp
        # timestamp -= first
        timestamps.append(timestamp)

    return timestamps, images

minute = 43

plt.title("Timestamp vs index")
cam_path = Path(f"/home/forge/ws/src/libuavf_2024/scripts/test_image_log")
timestamps, imgs = get_images(cam_path)
plt.scatter(range(len(timestamps)), timestamps, s=1)

# img_process_path = Path(f"/home/ericp/uavf_2024/flight_logs/perception_logs_608/06-08 01:{minute}/image_processor")
# for folder in sorted(img_process_path.glob("img_*")):
#     data = json.load(open(folder / "data.json"))
#     timestamp = data['image_time']
#     plt.plot([0, len(timestamps)], [timestamp, timestamp], color='red', linewidth=1)

plt.legend()
plt.savefig("timestamps_vs_index.png")