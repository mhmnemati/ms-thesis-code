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
    def __init__(self, n_times, n_outputs):
        super().__init__()
        self.conv1 = G.GATConv(in_channels=n_times, out_channels=int(n_times/2))
        self.conv2 = G.GATConv(in_channels=int(n_times/2), out_channels=int(n_times/4))
        self.linear = T.Linear(in_features=int(n_times/4), out_features=n_outputs)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = G.global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        return x


class Brain2Vec(BaseModel):
    def __init__(self):
        super().__init__(
            get_model=lambda: Model(n_times=100, n_outputs=2),
            get_loss=F.cross_entropy,
            num_classes=2
        )

    @staticmethod
    def transform(item):
        data = item["data"]
        sources = item["sources"]
        targets = item["targets"]

        electrodes = np.unique(np.concatenate([sources, targets]), axis=0)

        node_features = np.zeros((electrodes.shape[0], data.shape[1]), dtype=np.float32)
        for i in range(len(data)):
            power = data[i] ** 2

            source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
            target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

            node_features[source_idx] += power / 2
            node_features[target_idx] += power / 2

        adjecancy_matrix = np.zeros((electrodes.shape[0], electrodes.shape[0]), dtype=np.float64)
        for i in range(electrodes.shape[0]):
            for j in range(electrodes.shape[0]):
                distance = np.linalg.norm(electrodes[j] - electrodes[i])
                adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0

        edge_index = from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0]

        return Data(x=torch.from_numpy(node_features), edge_index=edge_index, y=torch.tensor([item["label"]], dtype=torch.int64))
