import os
import pywt
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

        self.model = G.Sequential("x, edge_index, graph_size, graph_length, batch", [
            (T.InstanceNorm1d(num_features=int(n_times/1)), "x -> x"),

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
            (T.GRU(input_size=int(n_times/8), hidden_size=128, num_layers=3, bidirectional=True, dropout=0.3), "x -> x, h"),

            (T.Linear(in_features=256, out_features=n_outputs), "x -> x"),
            (T.Sigmoid(), "x -> x"),
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
            loss=F.binary_cross_entropy
        )

        self.edge_select = kwargs["edge_select"]
        self.wave_transform = kwargs["wave_transform"]

    def transform(self, item):
        data = item["data"]
        labels = item["labels"]
        sources = item["sources"]
        targets = item["targets"]
        ch_names = item["ch_names"]

        source_names = [name.replace("EEG ", "").split("-")[0] for name in ch_names]
        target_names = [name.replace("EEG ", "").split("-")[1] for name in ch_names]

        electrode_positions = np.concatenate([sources, targets])
        electrode_names = (source_names + target_names)
        electrodes = list(set(electrode_names))
        n_electrodes = len(electrodes)
        n_times = int(data.shape[1])

        node_features = np.zeros((n_electrodes, n_times), dtype=np.float32)
        for i in range(data.shape[0]):
            # Convert bipolar wave data to electrode node_features
            power = data[i, :]

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

        adjecancy_matrix = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)
        for i in range(n_electrodes):
            # Inter graph connections (const/cluster/dynamic/...)
            for j in range(n_electrodes):
                if self.edge_select == "far":
                    y = electrode_positions[electrode_names.index(electrodes[j])]
                    x = electrode_positions[electrode_names.index(electrodes[i])]
                    distance = np.linalg.norm(y - x)
                    adjecancy_matrix[i, j] = 1 if distance > 0.1 else 0
                elif self.edge_select == "close":
                    y = electrode_positions[electrode_names.index(electrodes[j])]
                    x = electrode_positions[electrode_names.index(electrodes[i])]
                    distance = np.linalg.norm(y - x)
                    adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0
                elif self.edge_select == "cluster":
                    data = self.distances
                    distance = data.loc[(data["from"] == f"EEG {electrodes[i]}") & (data["to"] == f"EEG {electrodes[j]}")]
                    if len(distance) > 0:
                        distance = distance.iloc[0]["distance"]
                        if distance > 0.9:
                            adjecancy_matrix[i, j] = 1
                elif self.edge_select == "dynamic":
                    # TODO: implementation needed
                    pass

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(labels.max()),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
            graph_size=pt.tensor(n_electrodes),
            graph_length=1,
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("GCNBiGRU")
        parser.add_argument("--n_times", type=int, default=256)
        parser.add_argument("--n_outputs", type=int, default=1)
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="raw", choices=["raw", "fourier", "wavelet"])
        return parent_parser
