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


if __name__ == '__main__':
    plt.title("Timestamp vs index")
    logs_path = Path("/home/forge/ws/logs/06-12 20h21m")
    cam_path = logs_path / "poses"
    timestamps, imgs = get_images(cam_path)
    
    first = timestamps[0]
    last = timestamps[-1]
    
    print(f"First timestamp: {first}. last timestamp: {last}. duration: {last - first}")
    
    plt.scatter(range(len(timestamps)), timestamps, s=1)

    # img_process_path = logs_path / "image_processor"
    # for folder in sorted(img_process_path.glob("img_*")):
    #     data = json.load(open(folder / "data.json"))
    #     timestamp = data['image_time']
    #     plt.plot([0, len(timestamps)], [timestamp, timestamp], color='red', linewidth=1)

    plt.legend()
    plt.savefig("timestamps_vs_index.png")