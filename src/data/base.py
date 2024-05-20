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
    def __init__(self, name, filters, transform, data_loader, num_workers, batch_size):
        super().__init__()
        self.root = f"~/pytorch_datasets/{name}"
        self.filters = filters
        self.transform = transform
        self.data_loader = data_loader
        self.num_workers = num_workers
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
        return self.data_loader(self.trainset, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.data_loader(self.validset, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.data_loader(self.testset, num_workers=self.num_workers, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.data_loader(self.predictset, num_workers=self.num_workers, batch_size=self.batch_size)
