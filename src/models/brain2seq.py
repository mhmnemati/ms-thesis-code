import torch
import numpy as np
import scipy as sp
import torch.nn as T
import torch_geometric.nn as G
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseModel


class Model(T.Module):
    def __init__(self, n_times, n_outputs, layer_type):
        super().__init__()
        Conv = G.GATConv
        if layer_type == "GCN":
            Conv = G.GCNConv
        elif layer_type == "GCN2":
            Conv = G.GCN2Conv
        elif layer_type == "GAT":
            Conv = G.GATConv
        elif layer_type == "GAT2":
            Conv = G.GATv2Conv
        elif layer_type == "Cheb":
            Conv = G.ChebConv

        self.model = G.Sequential("x, edge_index, batch", [
            (Conv(in_channels=n_times, out_channels=int(n_times/2)), "x, edge_index -> x"),
            (T.ReLU(), "x -> x"),
            (Conv(in_channels=int(n_times/2), out_channels=int(n_times/4)), "x, edge_index -> x"),
            (G.MeanAggregation(), "x, batch -> x"),
            (T.Dropout(p=0.5), "x -> x"),
            (T.Linear(in_features=int(n_times/4), out_features=n_outputs), "x -> x")
        ])

    def forward(self, x, edge_index, batch):
        return self.model(x, edge_index, batch)


class Brain2Seq(BaseModel):
    def __init__(self, n_times=100, n_outputs=2, layer_type="GCN"):
        super().__init__(
            num_classes=2,
            hparams={k: v for k, v in locals().items() if k not in ["self", "__class__"]},
            model=Model(n_times=n_times, n_outputs=n_outputs, layer_type=layer_type),
            loss=F.cross_entropy,
        )

    @staticmethod
    def transform(item):
        data = item["data"]
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

        return Data(x=torch.from_numpy(node_features), edge_index=edge_index, y=torch.tensor([item["label"]], dtype=torch.int64))
