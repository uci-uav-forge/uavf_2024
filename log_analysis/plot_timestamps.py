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
logs_path = Path("/media/forge/SANDISK/logs/06-11 21h01m/")
cam_path = logs_path / "camera"
timestamps, imgs = get_images(cam_path)
plt.scatter(range(len(timestamps)), timestamps, s=1)

# img_process_path = logs_path / "image_processor"
# for folder in sorted(img_process_path.glob("img_*")):
#     data = json.load(open(folder / "data.json"))
#     timestamp = data['image_time']
#     plt.plot([0, len(timestamps)], [timestamp, timestamp], color='red', linewidth=1)

plt.legend()
plt.savefig("timestamps_vs_index.png")