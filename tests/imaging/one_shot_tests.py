import time

import torch
from torchinfo import summary
from tqdm import tqdm

from uavf_2024.imaging.one_shot.one_shot import SUASYOLO
from memory_profiler import profile

USE_CUDA = True
device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

model = SUASYOLO(num_classes = 14).to(device)
summary(model, (1, 3, 640, 640))

# @profile
def test(num_repeats=1, batch_size=1):
    print(f"Testing batch size {batch_size}")
    start =  time.perf_counter()
    for _ in tqdm(range(num_repeats)):
        x = torch.randint(0,255,(batch_size, 3, 640, 640), dtype=torch.uint8)
        x=x.to(device).type(torch.float32)
        boxes, classes, objectness = model.predict(x)
    end = time.perf_counter()
    print(f"Time taken: {end-start} ({(end-start)/batch_size} per {num_repeats} images)")

@profile
def run_all_tests():
    test(batch_size=1)
    # test(batch_size=2)
    # test(batch_size=4)

run_all_tests()