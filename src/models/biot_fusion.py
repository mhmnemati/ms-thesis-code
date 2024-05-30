import numpy as np
import scipy as sp
import torch as pt
import torch.nn as nn

from .base import BaseModel
from torch.utils.data import DataLoader

from .biot import BIOTEncoder


class BIOTFusionModel(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.eeg_encoder = BIOTEncoder(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=5,
            n_fft=200,
            hop_length=100,
            n_channels=18,
        )
        self.exg_encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=64),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=16),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16),
            nn.MaxPool1d(kernel_size=4),

            nn.AvgPool1d(kernel_size=4),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256+128, out_features=n_outputs),
        )

    def forward(self, x):
        x1 = self.eeg_encoder(x["EEG"])
        x2 = self.exg_encoder(x["EXG"])
        return self.classifier(pt.cat((x1, x2), -1))


class BIOTFusion(BaseModel):
    data_loader = DataLoader

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=BIOTFusionModel(**hparams),
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

        data_eeg = np.zeros((len(channels), 30 * 200), dtype=np.float32)
        for idx, ch_name in enumerate(item["ch_names"]):
            ch_name = ch_name.replace("EEG ", "").upper()

            if ch_name in channels:
                signal = sp.signal.resample(item["data"][idx], 30 * 200)
                data_eeg[channels.index(ch_name)] = signal

            if ch_name == "FPZ-CZ":
                signal = sp.signal.resample(item["data"][idx], 30 * 200) / 2
                data_eeg[channels.index("FP1-F3")] = signal
                data_eeg[channels.index("F3-C3")] = signal
                data_eeg[channels.index("FP2-F4")] = signal
                data_eeg[channels.index("F4-C4")] = signal

            if ch_name == "PZ-OZ":
                signal = sp.signal.resample(item["data"][idx], 30 * 200)
                data_eeg[channels.index("P3-O1")] = signal
                data_eeg[channels.index("P4-O2")] = signal

        data_exg = item["data"][[
            idx for idx, ch_name in enumerate(item["ch_names"]) if "EEG" not in ch_name
        ]]

        return ({"EEG": data_eeg, "EXG": data_exg}, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("BIOTFusion")
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
