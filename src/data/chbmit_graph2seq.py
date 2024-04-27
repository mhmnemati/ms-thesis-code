import numpy as np
import scipy as sp
import torch as pt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseDataset
from .chbmit import Generator


class CHBMITGraph2Seq(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            generator=Generator(),
            transform=self.transform,
            data_loader=DataLoader,
            batch_size=kwargs["batch_size"],
        )

    def transform(self, item):
        data = item["data"]
        labels = item["labels"]
        sources = item["sources"]
        targets = item["targets"]

        electrodes = np.unique(np.concatenate([sources, targets]), axis=0)

        node_features = np.zeros((electrodes.shape[0], data.shape[1]), dtype=np.float32)
        for i in range(len(data)):
            # TODO: use other transformations (wavelet, fourier, hilbert, ...)
            power = data[i] ** 2

            source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
            target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

            node_features[source_idx] += power / 2
            node_features[target_idx] += power / 2

        adjecancy_matrix = np.zeros((electrodes.shape[0], electrodes.shape[0]), dtype=np.float64)
        for i in range(electrodes.shape[0]):
            for j in range(electrodes.shape[0]):
                # TODO: construct graph edges methods (constant, clustering, dynamic, ...)
                distance = np.linalg.norm(electrodes[j] - electrodes[i])
                adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0

        edge_index = from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0]
        y = pt.tensor(labels)

        return Data(x=pt.from_numpy(node_features), edge_index=edge_index, y=y)

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--batch_size", type=int, default=8)
        return parent_parser
