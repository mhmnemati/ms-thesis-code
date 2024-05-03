import numpy as np
import scipy as sp
import torch as pt

from torch_geometric.data import Data
from torch.utils.data import DataLoader as TensorDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseDataset


class CHBMIT(BaseDataset):
    def __init__(self, **kwargs):
        transform = self.vec2vec
        data_loader = TensorDataLoader
        if kwargs["batch_type"] == "graph2vec":
            transform = self.graph2vec
            data_loader = GraphDataLoader
        elif kwargs["batch_type"] == "graph2seq":
            transform = self.graph2seq
            data_loader = GraphDataLoader

        super().__init__(
            name="chb_mit_window_30_overlap_5",
            transform=transform,
            data_loader=data_loader,
            batch_size=kwargs["batch_size"],
        )

    def vec2vec(self, item):
        data = item["data"]         # (23, 3000)
        labels = item["labels"]     # (30,)

        y = pt.tensor(labels.max())

        return (data, y)

    def graph2vec(self, item):
        data = item["data"]         # (23, 3000)
        labels = item["labels"]     # (30,)
        sources = item["sources"]   # (23, 3)
        targets = item["targets"]   # (23, 3)
        # electrodes    (21, 3)
        # node_feature  (21, 3000)
        # edge_index    (21, 21)
        # y             (1)

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
        y = pt.tensor(labels.max())

        return Data(x=pt.from_numpy(node_features), edge_index=edge_index, y=y)

    def graph2seq(self, item):
        data = item["data"]         # (23, 3000)
        labels = item["labels"]     # (30,)
        sources = item["sources"]   # (23, 3)
        targets = item["targets"]   # (23, 3)
        # electrodes    (21, 3)
        # node_feature  (21*30, 3000/30)
        # edge_index    (21*30, 21*30)
        # y             (30)

        electrodes = np.unique(np.concatenate([sources, targets]), axis=0)
        length = labels.shape[0]
        elecs = electrodes.shape[0]

        node_features = np.zeros((electrodes.shape[0] * length, int(data.shape[1] / length)), dtype=np.float32)
        for x in range(length):
            for i in range(data.shape[0]):
                # TODO: use other transformations (wavelet, fourier, hilbert, ...)
                part = slice(x*100, (x+1)*100)
                power = data[i, part] ** 2

                source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
                target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

                node_features[source_idx] += power / 2
                node_features[target_idx] += power / 2

        adjecancy_matrix = np.zeros((electrodes.shape[0] * length, electrodes.shape[0] * length), dtype=np.float64)
        for x in range(length):
            for i in range(electrodes.shape[0]):
                for c in range(1, 3):
                    if x + c < length:
                        adjecancy_matrix[(x*elecs)+i, ((x+c)*elecs)+i] = 1

                for j in range(electrodes.shape[0]):
                    # TODO: construct graph edges methods (constant, clustering, dynamic, ...)
                    distance = np.linalg.norm(electrodes[j] - electrodes[i])
                    adjecancy_matrix[(x*elecs)+i, (x*elecs)+j] = 1 if distance < 0.1 else 0

        edge_index = from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0]
        y = pt.tensor(labels.reshape(1, -1))

        return Data(x=pt.from_numpy(node_features), edge_index=edge_index, y=y)

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--batch_type", type=str, default="vec2vec", choices=["vec2vec", "graph2vec", "graph2seq"])
        return parent_parser
