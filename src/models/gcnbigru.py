import os
import pywt
import numpy as np
import scipy as sp
import torch as pt
import pandas as pd
import torch.nn as T
import focal_loss as fl
import torch_geometric.nn as G

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

            (G.MinAggregation(), "x, batch -> x"),
            (T.MultiheadAttention(embed_dim=int(n_times/8), num_heads=8, dropout=0.3), "x, x, x -> x, _"),

            (T.Linear(in_features=int(n_times/8), out_features=n_outputs), "x -> x"),
            (T.Softmax(dim=-1), "x -> x"),
        ])

    def forward(self, *args):
        return self.model(*args)


class GCNBiGRU(BaseModel):
    data_loader = DataLoader
    distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=fl.FocalLoss(gamma=0.7),
        )

        self.edge_select = kwargs["edge_select"]
        self.wave_transform = kwargs["wave_transform"]

    def transform(self, item):
        source_names = [name.replace("EEG ", "").split("-")[0] for name in item["ch_names"]]
        target_names = [name.replace("EEG ", "").split("-")[1] for name in item["ch_names"]]
        # electrode_positions = np.concatenate([item["sources"], item["targets"]])
        electrode_names = (source_names + target_names)
        electrodes = list(set(electrode_names))

        node_features = np.zeros((len(electrodes), item["data"].shape[1]), dtype=np.float32)
        for i in range(node_features.shape[0]):
            neighbours = [idx for idx, ch_name in enumerate(item["ch_names"]) if electrodes[i] in ch_name]
            weights = item["sources"][neighbours] - item["targets"][neighbours]
            signals = item["data"][neighbours]

        for i in range(item["data"].shape[0]):
            # Convert bipolar wave data to electrode node_features
            power = item["data"][i]

            if self.wave_transform == "raw":
                power = power ** 2
            elif self.wave_transform == "fourier":
                power = np.abs(np.fft.fft(power)) ** 2
            elif self.wave_transform == "wavelet":
                coeffs = pywt.wavedec(power, "db4", level=5)
                coeffs[-1] = np.zeros_like(coeffs[-1])
                coeffs[-2] = np.zeros_like(coeffs[-2])
                power = pywt.waverec(coeffs, "db4") ** 2

            node_features[electrodes.index(source_names[i])] += power / 2
            node_features[electrodes.index(target_names[i])] += power / 2

        adjecancy_matrix = np.zeros((node_features.shape[0], node_features.shape[0]), dtype=np.float64)
        for i in range(node_features.shape[0]):
            # Inter graph connections (const/cluster/dynamic/...)
            for j in range(node_features.shape[0]):
                if self.edge_select == "far":
                    x = electrode_positions[electrode_names.index(electrodes[i])]
                    y = electrode_positions[electrode_names.index(electrodes[j])]
                    distance = np.linalg.norm(y - x)
                    if distance > 0.1:
                        adjecancy_matrix[i, j] = 1
                elif self.edge_select == "close":
                    x = electrode_positions[electrode_names.index(electrodes[i])]
                    y = electrode_positions[electrode_names.index(electrodes[j])]
                    distance = np.linalg.norm(y - x)
                    if distance < 0.1:
                        adjecancy_matrix[i, j] = 1
                elif self.edge_select == "cluster":
                    distance = self.distances.loc[(self.distances["from"] == f"EEG {electrodes[i]}") & (self.distances["to"] == f"EEG {electrodes[j]}")]
                    if len(distance) > 0 and distance.iloc[0]["distance"] > 0.9:
                        adjecancy_matrix[i, j] = 1
                elif self.edge_select == "dynamic":
                    # TODO: implementation needed
                    pass

        for i in range(node_features.shape[0]):
            percentile_95 = np.percentile(np.abs(node_features[i]), 95, axis=0, keepdims=True)
            node_features[i] = node_features[i] / percentile_95

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(item["labels"].max()),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("GCNBiGRU")
        parser.add_argument("--n_times", type=int, default=256)
        parser.add_argument("--n_outputs", type=int, default=2)
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="raw", choices=["raw", "fourier", "wavelet"])
        return parent_parser
