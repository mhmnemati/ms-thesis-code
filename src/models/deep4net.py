import numpy as np
import torch as pt
import braindecode.models as M

from .base import BaseModel
from torch.utils.data import DataLoader


class Deep4Net(BaseModel):
    data_loader = DataLoader

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_chans", "n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=M.Deep4Net(**hparams),
            loss=pt.nn.CrossEntropyLoss(weight=pt.tensor([0.01, 10000]))
        )

    def transform(self, item):
        data = item["data"]
        label = item["labels"].max()

        if data.shape[0] != self.hparams.n_chans:
            data = np.concatenate([data, np.zeros((self.hparams.n_chans - data.shape[0], data.shape[1]))])

        return data, label

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Deep4Net")
        parser.add_argument("--n_chans", type=int, default=23)
        parser.add_argument("--n_times", type=int, default=3000)
        parser.add_argument("--n_outputs", type=int, default=2)
        return parent_parser
