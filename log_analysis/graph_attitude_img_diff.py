import json
from pathlib import Path
from typing import Iterable

import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def get_attitudes(poses_dir: Path, idx_range):
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    
    for path in tqdm(sorted(poses_dir.glob("*.json"))[idx_range[0]:idx_range[1]]):
        with open(path, "r") as f:
            datum = json.load(f)
            
        timestamps.append(datum["timestamp"])
            
        
        poses.append(np.array(datum['attitude']))
    
    print(f"Got {len(poses)} poses.")
    return timestamps, poses
        

def get_images(images_dir: Path, idx_range):
    timestamps: list[float] = []
    images: list[np.ndarray] = []
    
    for path in tqdm(sorted(images_dir.glob("*.jpg"))[idx_range[0]:idx_range[1]]):
        timestamp = float(path.stem)
        image = np.array(cv2.imread(str(path)))
        
        timestamps.append(timestamp)
        images.append(image)

    print(f"Got {len(images)} images.")
    return timestamps, images


def squared_diff(items: list[np.ndarray]):
    diffs: list[float] = [0]    
    
    for i in range(len(items) - 1):
        diff = np.sum(np.power(items[i] - items[i + 1], 2))
        diffs.append(diff)
        
    return diffs


def plot(
    pose_timestamps: list[float],
    pose_diffs: list[float],
    img_timestamps: list[float],
    img_diffs: list[float],
    save_path: Path = Path("img_vs_pose_diff.png")
):
    plt.xticks(np.arange(int(min(pose_timestamps)), int(max(pose_timestamps)), 0.2), rotation=90, fontsize=5)
    plt.plot(pose_timestamps, pose_diffs, color="blue", label="pose delta")
    plt.plot(img_timestamps, img_diffs, color="red", label="image delta")
    
    plt.xlabel("Time")
    plt.ylabel("Normalized Squared element-wise difference")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)


def normalize(data: Iterable[float]):
    arr = np.array(data)
    return (arr - np.mean(arr)) / np.max(arr)


def main():
    LOG_DIR = Path("../flight_logs/06-08 00:03").resolve()
    idx_range = (100, 300)
    pose_timestamps, poses = get_attitudes(LOG_DIR / "camera", idx_range)
    img_timestamps, images = get_images(LOG_DIR / "camera", idx_range)
    
    pose_diffs = normalize(squared_diff(poses))
    img_diffs = normalize(squared_diff(images))
    
    plot(pose_timestamps, pose_diffs, img_timestamps, img_diffs)

if __name__ == '__main__':
    main()
