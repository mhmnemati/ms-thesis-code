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

    def forward(self, x, modality_ids, context):
        x = self.encoder(x, modality_ids, context)
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
        ch_names = item["ch_names"]
        eeg_ch_indexes = [idx for idx, ch_name in enumerate(ch_names) if "EEG" in ch_name]
        eog_ch_indexes = [idx for idx, ch_name in enumerate(ch_names) if "EOG" in ch_name]

        for i in range(item["data"].shape[0]):
            percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            item["data"][i] = item["data"][i] / percentile_95

        eeg = item["data"][eeg_ch_indexes]
        eog = item["data"][eog_ch_indexes]

        sources_eeg = item["sources"][eeg_ch_indexes]
        targets_eeg = item["targets"][eeg_ch_indexes]

        sources_eog = item["sources"][eog_ch_indexes]
        targets_eog = item["targets"][eog_ch_indexes]

        positions_eeg = (sources_eeg + targets_eeg) / 2
        positions_eog = (sources_eog + targets_eog) / 2

        x = np.vstack((eeg, eog))
        modality_ids = np.repeat([0, 1], [len(eeg), len(eog)])
        context = np.vstack((positions_eeg, positions_eog))

        # Only in ISRUC
        if x.shape[0] != 6:
            for i in range(6 - x.shape[0]):
                x = np.vstack((x, x[[0]]))
                modality_ids = np.append(modality_ids, modality_ids[0])
                context = np.vstack((context, context[[0]]))

        return (x, modality_ids, context, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("BIOTRaw")
        parser.add_argument("--n_times", type=int, default=6000)
        parser.add_argument("--n_channels", type=int, default=4)
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
