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
        transform = self.tensor2vec
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

    def tensor2vec(self, item):
        return (item["data"], item["labels"].max())

    def get_graph(self, full, data, labels, sources, targets):
        # electrodes        (21, 3)

        # node_feature      (21, 3000)
        # adjacency_matrix  (21, 21)
        # y                 (1)

        # node_feature      (21*30, 3000/30)
        # adjacency_matrix  (21*30, 21*30)
        # y                 (30)

        electrodes = np.unique(np.concatenate([sources, targets]), axis=0)
        n_electrodes = electrodes.shape[0]
        n_graphs = 1 if (full == True) else labels.shape[0]
        n_times = int(data.shape[1] / n_graphs)

        node_features = np.zeros((n_electrodes * n_graphs, n_times), dtype=np.float32)
        for idx in range(n_graphs):
            for i in range(data.shape[0]):
                # TODO: use other transformations (wavelet, fourier, hilbert, ...)
                power = data[i, idx*n_times:(idx+1)*n_times] ** 2

                source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
                target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

                node_features[source_idx] += power / 2
                node_features[target_idx] += power / 2

        adjecancy_matrix = np.zeros((n_electrodes * n_graphs, n_electrodes * n_graphs), dtype=np.float64)
        for idx in range(n_graphs):
            for i in range(n_electrodes):
                # Cross graph connections (before,after)
                # TODO: parametric cross connections length
                for c in range(1, 3):
                    if idx + c < n_graphs:
                        adjecancy_matrix[(idx*n_electrodes)+i, ((idx+c)*n_electrodes)+i] = 1

                # Inter graph connections (const/cluster/dynamic/...)
                # TODO: construct graph edges methods (constant, clustering, dynamic, ...)
                for j in range(n_electrodes):
                    distance = np.linalg.norm(electrodes[j] - electrodes[i])
                    adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance < 0.1 else 0

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(labels.max() if (full == True) else labels.reshape(1, -1)),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
            graph_size=pt.tensor(n_electrodes),
            graph_length=pt.tensor(n_graphs),
        )

    def graph2vec(self, item):
        return self.get_graph(True, item["data"], item["labels"], item["sources"], item["targets"])

    def graph2seq(self, item):
        return self.get_graph(False, item["data"], item["labels"], item["sources"], item["targets"])

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--batch_type", type=str, default="tensor2vec", choices=["tensor2vec", "graph2vec", "graph2seq"])
        return parent_parser
