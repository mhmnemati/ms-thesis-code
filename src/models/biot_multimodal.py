import numpy as np
import torch as pt
import torch.nn as nn

from .base import BaseModel
from torch.utils.data import DataLoader

from .biot import BIOTEncoder
from linear_attention_transformer import LinearAttentionTransformer


class BIOTMultiModalModel(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super().__init__()

        self.eeg_encoder = BIOTEncoder(n_channels=2, heads=2, depth=4, n_fft=400)
        self.eog_encoder = BIOTEncoder(n_channels=1, heads=2, depth=2, n_fft=400)
        self.pre_transformer = nn.Sequential(
            nn.Dropout(p=0.5),
            # nn.LayerNorm(normalized_shape=),
        )
        self.x_transformer = LinearAttentionTransformer(dim=128, heads=2, depth=2, max_seq_len=128)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.GELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.GELU(),
            nn.Linear(in_features=32, out_features=n_outputs),
        )

    def forward(self, eeg, eog):
        x_eeg = self.eeg_encoder(eeg)
        x_eog = self.eog_encoder(eog)
        x = pt.cat([x_eeg, x_eog], dim=1)
        x1 = self.pre_transformer(x)
        x2 = self.x_transformer(x1[None, :, :])[0]
        x = x1 + x2
        return self.classifier(x)


class BIOTMultiModal(BaseModel):
    data_loader = DataLoader

    def __init__(self, **hparams):
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=BIOTMultiModalModel(
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

        return (eeg, eog, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("BIOTMultiModal")
        parser.add_argument("--n_times", type=int, default=6000)
        parser.add_argument("--n_channels", type=int, default=4)
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
