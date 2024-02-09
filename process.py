import numpy as np
from data import SleepEDFX

dataset = SleepEDFX(split="train")

for idx, val in enumerate(iter(dataset)):
    
