import os
import glob
import zipfile

from torch import save, load
from torch.hub import download_url_to_file
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root, split, transform, generators, url):
        super().__init__()
        self.transform = transform
        os.makedirs(root, exist_ok=True)

        # Download
        if not os.path.exists(f"{root}/data.zip"):
            download_url_to_file(url, f"{root}/data.zip")

        # Extract
        if not os.path.exists(f"{root}/extract"):
            with zipfile.ZipFile(f"{root}/data.zip", "r") as fd:
                fd.extractall(f"{root}/extract")

        # Transform
        if not os.path.exists(f"{root}/transform"):
            for key, records in generators(f"{root}/extract").items():
                path = f"{root}/transform/{key}"
                os.makedirs(path, exist_ok=True)

                for idx, item in enumerate(records):
                    save(item, f"{path}/{idx}.pt")

        # Initialize
        self.items = glob.glob(f"{root}/transform/{split}/*.pt")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = load(self.items[index])

        if self.transform:
            item = self.transform(item)

        return item
