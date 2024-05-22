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
    def __init__(self, n_times, n_outputs, layer_type, aggregator):
        super().__init__()
        Conv = G.GCNConv
        if layer_type == "gcn":
            Conv = G.GCNConv
        elif layer_type == "gcn2":
            Conv = G.GCN2Conv
        elif layer_type == "gat":
            Conv = G.GATConv
        elif layer_type == "gat2":
            Conv = G.GATv2Conv
        elif layer_type == "cheb":
            Conv = G.ChebConv

        Agg = G.MinAggregation
        if aggregator == "min":
            Agg = G.MinAggregation
        elif aggregator == "max":
            Agg = G.MaxAggregation
        elif aggregator == "mean":
            Agg = G.MeanAggregation
        elif aggregator == "median":
            Agg = G.MedianAggregation

        def batch_convert(graph_size, graph_length):
            repeats = pt.repeat_interleave(graph_size, graph_length)
            values = pt.full((graph_length.sum(), ), len(graph_length))

            i = 0
            for idx, length in enumerate(graph_length):
                values[i + int(length/2)] = idx
                i += length

            return pt.repeat_interleave(values, repeats)

        self.model = G.Sequential("x, edge_index, graph_size, graph_length, batch", [
            (Conv(in_channels=n_times, out_channels=int(n_times/2)), "x, edge_index -> x"),
            (T.ReLU(), "x -> x"),
            (Conv(in_channels=int(n_times/2), out_channels=int(n_times/4)), "x, edge_index -> x"),

            # batch = [0[3*5], 1[2*4], 2[1*3]]
            # batch_size = 3
            # graph_size = [3,2,1]
            # graph_length = [5,4,3]
            # batch_old = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2] = (26)
            # batch_new = [-,-,-,-,-,-,0,0,0,-,-,-,-,-,-, -,-,1,1,-,-,-,-, -,2,-] = (26)
            # Caution: this implementation is highly optimized and complex
            (batch_convert, "graph_size, graph_length -> batch"),
            (Agg(), "x, batch -> x"),
            (lambda x: x[:-1, :], "x -> x"),

            (T.Dropout(p=0.1), "x -> x"),
            (T.Linear(in_features=int(n_times/4), out_features=n_outputs), "x -> x"),
        ])

    def forward(self, *args):
        return self.model(*args)


class Brain2Seq(BaseModel):
    data_loader = DataLoader
    distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs", "layer_type", "aggregator"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=F.cross_entropy
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
        n_graphs = labels.shape[0]
        n_times = int(data.shape[1] / n_graphs)

        node_features = np.zeros((n_electrodes * n_graphs, n_times), dtype=np.float32)
        for idx in range(n_graphs):
            for i in range(data.shape[0]):
                # Convert bipolar wave data to electrode node_features
                power = data[i, idx*n_times:(idx+1)*n_times]

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

        adjecancy_matrix = np.zeros((n_electrodes * n_graphs, n_electrodes * n_graphs), dtype=np.float64)
        for idx in range(n_graphs):
            for i in range(n_electrodes):
                # Cross graph connections (before,after)
                # TODO: parametric cross connections length
                for c in range(1, 3):
                    if idx + c < n_graphs:
                        adjecancy_matrix[(idx*n_electrodes)+i, ((idx+c)*n_electrodes)+i] = 1

                # Inter graph connections (const/cluster/dynamic/...)
                for j in range(n_electrodes):
                    if self.edge_select == "far":
                        y = electrode_positions[electrode_names.index(electrodes[j])]
                        x = electrode_positions[electrode_names.index(electrodes[i])]
                        distance = np.linalg.norm(y - x)
                        adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance > 0.1 else 0
                    elif self.edge_select == "close":
                        y = electrode_positions[electrode_names.index(electrodes[j])]
                        x = electrode_positions[electrode_names.index(electrodes[i])]
                        distance = np.linalg.norm(y - x)
                        adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance < 0.1 else 0
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
            graph_length=pt.tensor(n_graphs),
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Brain2Seq")
        parser.add_argument("--n_times", type=int, default=100)
        parser.add_argument("--n_outputs", type=int, default=2)
        parser.add_argument("--layer_type", type=str, default="gcn", choices=["gcn", "gcn2", "gat", "gat2", "cheb"])
        parser.add_argument("--aggregator", type=str, default="min", choices=["min", "max", "mean", "median"])
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="raw", choices=["raw", "fourier", "wavelet"])
        return parent_parser
