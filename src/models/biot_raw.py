import numpy as np
import torch as pt
import torch.nn as nn

from .base import BaseModel
from torch.utils.data import DataLoader

from .biot import BIOTEncoder


class BIOTRawModel(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super().__init__()

        self.encoder = BIOTEncoder(
            emb_size=256,
            heads=8,
            depth=4,
            n_fft=200,
            hop_length=100,
            n_channels=n_channels,
            n_classes=n_outputs,
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=n_outputs),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class BIOTRaw(BaseModel):
    data_loader = DataLoader

    def __init__(self, **hparams):
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=BIOTRawModel(
                n_channels=hparams["n_channels"],
                n_outputs=hparams["n_outputs"]
            ),
            loss=pt.nn.CrossEntropyLoss()
        )

        self.n_times = hparams["n_times"]
        self.n_channels = hparams["n_channels"]

    def transform(self, item):
        for i in range(item["data"].shape[0]):
            percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            item["data"][i] = item["data"][i] / percentile_95

        return (item["data"], item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("BIOTRaw")
        parser.add_argument("--n_times", type=int, default=6000)
        parser.add_argument("--n_channels", type=int, default=4)
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
