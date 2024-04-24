import os
import glob
import zipfile

from torch import save, load
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from lightning import LightningDataModule


class TensorDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.items = glob.glob(f"{root}/*.pt")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = load(self.items[index])
        if self.transform:
            item = self.transform(item)

        return item


class BaseDataset(LightningDataModule):
    def __init__(self, name, url, generator, transform, data_loader, batch_size):
        super().__init__()
        self.root = os.path.expanduser(f"~/pytorch_datasets/{name}")
        self.url = url
        self.generator = generator
        self.transform = transform
        self.data_loader = data_loader
        self.batch_size = batch_size

    def prepare_data(self):
        os.makedirs(self.root, exist_ok=True)

        # Download
        if not os.path.exists(f"{self.root}/data.zip"):
            download_url_to_file(self.url, f"{self.root}/data.zip")

        # Extract
        if not os.path.exists(f"{self.root}/extract"):
            with zipfile.ZipFile(f"{self.root}/data.zip", "r") as fd:
                fd.extractall(f"{self.root}/extract")

        # Transform
        if not os.path.exists(f"{self.root}/transform"):
            for key, records in self.generator(f"{self.root}/extract").items():
                path = f"{self.root}/transform/{key}"
                os.makedirs(path, exist_ok=True)

                for idx, item in enumerate(records):
                    save(item, f"{path}/{idx}.pt")

    def setup(self, stage):
        if stage == "fit":
            self.trainset = TensorDataset(f"{self.root}/transform/train", transform=self.transform)
            self.validset = TensorDataset(f"{self.root}/transform/valid", transform=self.transform)

        if stage == "test":
            self.testset = TensorDataset(f"{self.root}/transform/test", transform=self.transform)

        if stage == "predict":
            self.predictset = TensorDataset(f"{self.root}/transform/predict", transform=self.transform)

    def train_dataloader(self):
        return self.data_loader(self.trainset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def val_dataloader(self):
        return self.data_loader(self.validset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def test_dataloader(self):
        return self.data_loader(self.testset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))

    def predict_dataloader(self):
        return self.data_loader(self.predictset, batch_size=self.batch_size, num_workers=int(self.batch_size/2))
