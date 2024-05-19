import os
import glob

from torch import load
from torch.utils.data import Dataset
from lightning import LightningDataModule


class TensorDataset(Dataset):
    def __init__(self, root, filter, transform):
        self.transform = transform
        self.items = filter(glob.glob(f"{os.path.expanduser(root)}/**/*.pt", recursive=True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = load(self.items[index])
        if self.transform:
            item = self.transform(item)

        return item


class BaseDataset(LightningDataModule):
    def __init__(self, name, filters, transform, data_loader, batch_size):
        super().__init__()
        self.root = f"~/pytorch_datasets/{name}"
        self.filters = filters
        self.transform = transform
        self.data_loader = data_loader
        self.batch_size = batch_size

    def setup(self, stage):
        if stage == "fit":
            self.trainset = TensorDataset(f"{self.root}/train", filter=self.filters("train"), transform=self.transform)
            self.validset = TensorDataset(f"{self.root}/train", filter=self.filters("valid"), transform=self.transform)

        if stage == "test":
            self.testset = TensorDataset(f"{self.root}/test", filter=self.filters("test"), transform=self.transform)

        if stage == "predict":
            self.predictset = TensorDataset(f"{self.root}/predict", filter=self.filters("predict"), transform=self.transform)

    def train_dataloader(self):
        return self.data_loader(self.trainset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def val_dataloader(self):
        return self.data_loader(self.validset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def test_dataloader(self):
        return self.data_loader(self.testset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def predict_dataloader(self):
        return self.data_loader(self.predictset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))
