import os
import glob
import numpy as np

from torch import load
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    seed = 100

    def __init__(self, name, split, folds, k, transform):
        root = os.path.expanduser(f"~/pytorch_datasets/{name}/{split}")

        parts = glob.glob(f"{root}/*")
        np.random.seed(self.seed)
        np.random.shuffle(parts)

        items = glob.glob(f"{root}/**/*.pt", recursive=True)
        np.random.seed(self.seed)
        np.random.shuffle(items)

        if k > 0:
            part_shards = np.array_split(parts, folds)
            parts = part_shards[k-1]
        elif k < 0:
            part_shards = np.array_split(parts, folds)
            parts = np.concatenate(part_shards[:abs(k)-1] + part_shards[abs(k):])

        self.items = list(filter(lambda item: any([p in item for p in parts]), items))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = load(self.items[index])
        if self.transform:
            item = self.transform(item)

        return item
