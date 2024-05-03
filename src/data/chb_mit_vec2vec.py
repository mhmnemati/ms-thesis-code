import torch as pt

from torch.utils.data import DataLoader

from .base import BaseDataset


class CHBMITVec2Vec(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            name="chb_mit_window_30_overlap_5",
            transform=self.transform,
            data_loader=DataLoader,
            batch_size=kwargs["batch_size"],
        )

    def transform(self, item):
        data = item["data"]         # (23, 3000)
        labels = item["labels"]     # (30,)

        return (data, labels.max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMITVec2Vec")
        parser.add_argument("--batch_size", type=int, default=8)
        return parent_parser
