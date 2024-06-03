import numpy as np
import scipy as sp
import torch as pt
import torch.nn as nn

from .base import BaseModel
from torch.utils.data import DataLoader

from .biot import BIOTEncoder


class BIOTRawModel(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.encoder = BIOTEncoder(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=5,
            n_fft=200,
            hop_length=100,
            n_channels=18,
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

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=BIOTRawModel(**hparams),
            loss=pt.nn.CrossEntropyLoss()
        )

    def transform(self, item):
        for i in range(item["data"].shape[0]):
            percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            item["data"][i] = item["data"][i] / percentile_95

        channels = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "C3-A2",
            "C4-A1",
        ]

        size = int(item["data"].shape[1] / 200)
        data = np.zeros((len(channels), size * 200), dtype=np.float32)
        for idx, ch_name in enumerate(item["ch_names"]):
            ch_name = ch_name.replace("EEG ", "").upper()

            if ch_name in channels:
                signal = sp.signal.resample(item["data"][idx], size * 200)
                data[channels.index(ch_name)] = signal

            if ch_name == "FPZ-CZ":
                signal = sp.signal.resample(item["data"][idx], size * 200) / 2
                data[channels.index("FP1-F3")] = signal
                data[channels.index("F3-C3")] = signal
                data[channels.index("FP2-F4")] = signal
                data[channels.index("F4-C4")] = signal

            if ch_name == "PZ-OZ":
                signal = sp.signal.resample(item["data"][idx], size * 200)
                data[channels.index("P3-O1")] = signal
                data[channels.index("P4-O2")] = signal

        return (data, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("BIOTRaw")
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
