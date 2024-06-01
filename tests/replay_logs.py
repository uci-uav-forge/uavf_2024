from uavf_2024.imaging.imaging_types import ProbabilisticTargetDescriptor, Target3D, CertainTargetDescriptor
from uavf_2024.imaging import TargetTracker
import numpy as np
import os
import json

if __name__=="__main__":
    root_folder = "imaging/2024_test_data/arc_test_530"
    tracker = TargetTracker()
    for frame_folder in os.listdir(root_folder):
        data = json.load(open(os.path.join(root_folder, frame_folder, "data.json")))
        preds_3d_dicts = data["preds_3d"] 
        for p in preds_3d_dicts:
            det_id = p['id'].split('/')[1]
            file_contents = open(f"{root_folder}/{frame_folder}/{det_id}/descriptor.txt").read()
            tracker.update([Target3D(
                np.array(p['position']),
                ProbabilisticTargetDescriptor.from_string(file_contents),
                id = f"{frame_folder}/{det_id}"
            )])

    targets = [
        CertainTargetDescriptor('red','pentagon','orange','F'),
        CertainTargetDescriptor('red','rectangle','white','T'),
        CertainTargetDescriptor('black','pentagon','white','G'),
        CertainTargetDescriptor('green','triangle','white','7'),
        CertainTargetDescriptor('purple','semicircle','blue','R')
    ]

    positions = tracker.estimate_positions(targets)

    for target, pos in zip(targets, positions):
        print(f"Target {target} is at position {pos}")
        print(f"Contributing measurements: {pos.contributing_measurement_ids()}")
        


