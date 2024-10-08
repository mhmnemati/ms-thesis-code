import numpy as np
import torch as pt
import braindecode.models as M

from .base import BaseModel
from torch.utils.data import DataLoader


class EEGInception(BaseModel):
    data_loader = DataLoader

    def __init__(self, **hparams):
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=M.EEGInception(
                n_chans=hparams["n_chans"],
                n_times=hparams["n_times"],
                n_outputs=hparams["n_outputs"],
            ),
            loss=pt.nn.CrossEntropyLoss()
        )

    def transform(self, item):
        for i in range(item["data"].shape[0]):
            percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            item["data"][i] = item["data"][i] / percentile_95

        data = np.zeros((self.hparams.n_chans, item["data"].shape[1]), dtype=np.float32)
        data[:item["data"].shape[0]] = item["data"]

        return data, item["labels"].max()

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("EEGInception")
        parser.add_argument("--n_chans", type=int, default=23)
        parser.add_argument("--n_times", type=int, default=3000)
        parser.add_argument("--n_outputs", type=int, default=2)
        return parent_parser
