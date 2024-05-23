import os
import numpy as np
import scipy as sp
import torch as pt
import pandas as pd
import torch.nn as T
import torch_geometric.nn as G
import torch.nn.functional as F

from .base import BaseModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class Model(T.Module):
    def __init__(self, n_times, n_outputs):
        super().__init__()

        self.model = G.Sequential("x, edge_index, batch", [
            (G.GCNConv(in_channels=int(n_times/1), out_channels=int(n_times/2)), "x, edge_index -> x"),
            (T.BatchNorm1d(num_features=int(n_times/2)), "x -> x"),
            (T.ReLU(), "x -> x"),
            (G.GCNConv(in_channels=int(n_times/2), out_channels=int(n_times/4)), "x, edge_index -> x"),
            (T.BatchNorm1d(num_features=int(n_times/4)), "x -> x"),
            (T.ReLU(), "x -> x"),
            (G.GCNConv(in_channels=int(n_times/4), out_channels=int(n_times/8)), "x, edge_index -> x"),
            (T.BatchNorm1d(num_features=int(n_times/8)), "x -> x"),
            (T.ReLU(), "x -> x"),

            (G.MeanAggregation(), "x, batch -> x"),
            (T.MultiheadAttention(embed_dim=int(n_times/8), num_heads=8, dropout=0.3), "x, x, x -> x, _"),

            (T.Linear(in_features=int(n_times/8), out_features=n_outputs), "x -> x"),
            (T.Sigmoid(), "x -> x"),
        ])

    def forward(self, *args):
        return self.model(*args)


class GCNAttn(BaseModel):
    data_loader = DataLoader
    distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=F.binary_cross_entropy
        )

    def transform(self, item):
        data = item["data"]
        labels = item["labels"]

        for i in range(data.shape[0]):
            percentile_95 = np.percentile(np.abs(data[i]), 95, axis=0, keepdims=True)
            data[i] = data[i] / percentile_95

        node_features = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        for i in range(data.shape[0]):
            node_features[i] = np.abs(np.fft.fft(data[i, :]))

        adjecancy_matrix = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                correlation = sp.stats.pearsonr(data[i], data[j]).statistic
                if correlation > 0.3:
                    adjecancy_matrix[i, j] = 1

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(labels.max()),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("GCNAttn")
        parser.add_argument("--n_times", type=int, default=256)
        parser.add_argument("--n_outputs", type=int, default=1)
        return parent_parser
