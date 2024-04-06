import os

import torch
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, DataLoader


class CHBMITDataset(Dataset):
    sfreq = 100
    window = 30
    overlap = 0
    labels = [
        "Normal",
        "Seizure"
    ]

    def __init__(self, root="./data/chb-mit"):
        super().__init__()
        os.makedirs(root, exist_ok=True)

        if not os.path.exists(f"{root}/dataset.zip"):
            # Download
            download_url_to_file("https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip", f"{root}/dataset.zip")

        if not os.path.exists(f"{root}/extract"):
            # Extract
            pass

        if not os.path.exists(f"{root}/process"):
            # Process
            pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return 1


CHBMITDataset()
