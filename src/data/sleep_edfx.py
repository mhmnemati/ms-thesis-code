import torch as pt

from torch.utils.data import DataLoader

from .base import BaseDataset


class SleepEDFX(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            name="sleep_edfx_window_30_overlap_5",
            transform=self.tensor2vec,
            data_loader=DataLoader,
            batch_size=kwargs["batch_size"],
        )

    def tensor2vec(self, item):
        data = item["data"]         # (6, 3000)
        labels = item["labels"]     # (30,)

        y = pt.tensor(labels.max())

        return (data, y)

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("SleepEDFX")
        parser.add_argument("--batch_size", type=int, default=8)
        return parent_parser
