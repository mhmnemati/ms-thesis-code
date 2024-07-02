import numpy as np
import torch as pt
import torch.nn as T

from .base import BaseModel
from torch.utils.data import DataLoader


class Model(T.Module):
    def __init__(self, n_chans, n_times, n_outputs):
        super().__init__()

        self.model = T.Sequential(
            T.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 4), padding="same"),
            T.ReLU(),
            T.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 4), stride=(1, 2)),
            T.ReLU(),
            T.MaxPool2d((1, 2)),

            T.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 4), padding="same"),
            T.ReLU(),
            T.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 4), stride=(1, 2)),
            T.ReLU(),
            T.MaxPool2d((2, 2)),

            T.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), padding="same"),
            T.ReLU(),
            T.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=(1, 2)),
            T.ReLU(),
            T.MaxPool2d((1, 2)),

            T.AdaptiveAvgPool2d((1, 1)),
            T.Flatten(),

            T.Linear(256, 256),
            T.ReLU(),
            T.Dropout(0.25),
            T.Linear(256, 128),
            T.ReLU(),
            T.Linear(128, 64),
            T.ReLU(),
            T.Dropout(0.25),
            T.Linear(64, 2),
        )

    def forward(self, x):
        return self.model(x)


class EEGCNN(BaseModel):
    data_loader = DataLoader

    def __init__(self, **hparams):
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(
                n_chans=hparams["n_chans"],
                n_times=hparams["n_times"],
                n_outputs=hparams["n_outputs"],
            ),
            loss=pt.nn.CrossEntropyLoss()
        )

    def transform(self, item):
        item["data"] = item["data"] * 1e6

        data = np.zeros((self.hparams.n_chans, item["data"].shape[1]), dtype=np.float32)
        data[:item["data"].shape[0]] = item["data"]

        return data[np.newaxis, :, :], item["labels"].max()

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("EEGCNN")
        parser.add_argument("--n_chans", type=int, default=23)
        parser.add_argument("--n_times", type=int, default=3000)
        parser.add_argument("--n_outputs", type=int, default=2)
        return parent_parser
