import os
import glob
import numpy as np

from torch import load
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    seed = 100

    def __init__(self, name, split, method, folds, k, transform):
        root = os.path.expanduser(f"~/pytorch_datasets/{name}/{split}")

        items = glob.glob(f"{root}/**/*.pt", recursive=True)
        np.random.seed(self.seed)
        np.random.shuffle(items)

        if method == "kfold":
            parts = glob.glob(f"{root}/*")
            np.random.seed(self.seed)
            np.random.shuffle(parts)

            if k > 0:
                part_shards = np.array_split(parts, folds)
                parts = part_shards[k-1]
            elif k < 0:
                part_shards = np.array_split(parts, folds)
                parts = np.concatenate(part_shards[:abs(k)-1] + part_shards[abs(k):])

            self.items = list(filter(lambda item: any([p in item for p in parts]), items))
        elif method == "labels":
            partitions = {}
            for item in items:
                segments = item.split("/")
                part_key = f"{segments[-3]}-{segments[-2]}"

                if part_key not in partitions:
                    partitions[part_key] = []

                partitions[part_key].append(item)

            self.items = []
            for items in partitions.values():
                item_shards = np.array_split(items, folds)

                if k > 0:
                    self.items += item_shards[k-1].tolist()
                elif k < 0:
                    self.items += np.concatenate(item_shards[:abs(k)-1] + item_shards[abs(k):]).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = load(self.items[index])
        if self.transform:
            item = self.transform(item)

        return item
