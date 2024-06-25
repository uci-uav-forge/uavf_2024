import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import os
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


plt.title(f"timestamp vs index")
logs_root_path = Path("/mnt/nvme/logs")
logs_path = logs_root_path / sorted([x for x in os.listdir(logs_root_path) if x.startswith("0")])[-1]
cam_path = logs_path / "camera"
timestamps, imgs = get_images(cam_path)
plt.scatter(range(len(timestamps)), timestamps, s=1, label="camera")
# pose_path = logs_path / "pose"
# timestamps, imgs = get_images(pose_path)
# plt.scatter(range(len(timestamps)), timestamps, s=1, label="pose")


img_process_path = logs_path / "image_processor"
for folder in sorted(img_process_path.glob("img_*")):
    try:
        data = json.load(open(folder / "data.json"))
        timestamp = data['image_time']
        plt.plot([0, len(timestamps)], [timestamp, timestamp], color='red', linewidth=1)
    except FileNotFoundError:
        print(f"No data file for {folder}")
        continue

plt.legend()
plt.savefig("timestamps_vs_index.png")