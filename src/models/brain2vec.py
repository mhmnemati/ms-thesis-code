import os
import pywt
import numpy as np
import scipy as sp
import torch as pt
import pandas as pd
import torch.nn as T
import focal_loss as fl
import torch_geometric.nn as G
import torch_geometric.utils as GU

from .base import BaseModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class Model(T.Module):
    def __init__(self, n_times, n_outputs, aggregator, gru_size):
        super().__init__()

        def batch_convert(n_nodes, n_graphs):
            repeats = pt.repeat_interleave(n_nodes, n_graphs)
            values = pt.full((n_graphs.sum(), ), len(n_graphs))

            i = 0
            for idx, length in enumerate(n_graphs):
                values[i + int(length/2)] = idx
                i += length

            return pt.repeat_interleave(values, repeats)

        self.model = G.Sequential("x, edge_index, n_nodes, n_graphs, batch", [
            (lambda edge_index: GU.dropout_edge(edge_index, p=0.2), "edge_index -> edge_index, edge_mask"),
            (G.GATv2Conv(in_channels=int(n_times*1), out_channels=int(n_times/4), heads=8, dropout=0.4), "x, edge_index -> x"),
            (T.BatchNorm1d(num_features=int(n_times*2)), "x -> x"),
            (T.ReLU(), "x -> x"),
            (G.GATv2Conv(in_channels=int(n_times*2), out_channels=int(n_times/2), heads=8), "x, edge_index -> x"),
            (T.BatchNorm1d(num_features=int(n_times*4)), "x -> x"),
            (T.ReLU(), "x -> x"),
            # (G.GATv2Conv(in_channels=int(n_times/1), out_channels=int(n_times/1)), "x, edge_index -> x"),
            # (T.BatchNorm1d(num_features=int(n_times/1)), "x -> x"),
            # (T.ReLU(), "x -> x"),

            (G.MeanAggregation(), "x, batch -> x"),

            # (T.MultiheadAttention(embed_dim=int(n_times/8), num_heads=2, dropout=0.3), "x, x, x -> x, _"),
            # (T.GRU(input_size=int(n_times*8), hidden_size=gru_size, num_layers=3, bidirectional=True, dropout=0.3), "x -> x, h"),

            (T.Linear(in_features=int(n_times*4), out_features=n_outputs), "x -> x"),
        ])

    def forward(self, *args):
        return self.model(*args)


class Brain2Vec(BaseModel):
    data_loader = DataLoader
    distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, **hparams):
        loss_fn = None
        if hparams["loss_fn"] == "ce":
            loss_fn = pt.nn.CrossEntropyLoss()
        elif hparams["loss_fn"] == "focal":
            loss_fn = fl.FocalLoss(gamma=0.7)

        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(
                n_times=hparams["n_times"],
                n_outputs=hparams["n_outputs"],
                aggregator=hparams["aggregator"],
                gru_size=hparams["gru_size"],
            ),
            loss=loss_fn,
        )

        self.cross_connections = hparams["cross_connections"]
        self.signal_transform = hparams["signal_transform"]
        self.node_transform = hparams["node_transform"]
        self.edge_select = hparams["edge_select"]
        self.threshold = hparams["threshold"]
        self.n_times = hparams["n_times"]

    def transform(self, item):
        # Signal Transform
        for i in range(item["data"].shape[0]):
            # Normalization: Method 1
            item["data"][i] = item["data"][i] * 1e6

            # Normalization: Method 2
            # for i in range(item["data"].shape[0]):
            #     percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            #     item["data"][i] = item["data"][i] / percentile_95

            # Normalization: Method 3
            # for i in range(item["data"].shape[0]):
            #     norm = np.linalg.norm(item["data"][i])
            #     item["data"][i] = item["data"][i] / norm

            if self.signal_transform == "fourier":
                coeffs = np.fft.fft(item["data"][i])
                coeffs[:2] = 0
                coeffs[33:] = 0
                item["data"][i] = np.real(np.fft.ifft(coeffs))
            elif self.signal_transform == "wavelet":
                coeffs = pywt.wavedec(item["data"][i], "db4", level=5)
                coeffs[-1] = np.zeros_like(coeffs[-1])
                coeffs[-2] = np.zeros_like(coeffs[-2])
                item["data"][i] = pywt.waverec(coeffs, "db4")

        n_graphs = int(item["data"].shape[1] / self.n_times)

        # Node Transform
        node_names = None
        node_positions = None
        node_features = None
        if self.node_transform == "unipolar":
            source_names = [name.replace("EEG ", "").split("-")[0] for name in item["ch_names"]]
            target_names = [name.replace("EEG ", "").split("-")[1] for name in item["ch_names"]]
            all_names = (source_names + target_names)
            node_names = list(set(all_names))

            all_positions = np.concatenate([item["sources"], item["targets"]])
            node_positions = np.zeros((len(node_names), 3), dtype=np.float32)
            for i in range(len(node_names)):
                node_positions[i] = all_positions[all_names.index(node_names[i])]

            node_features = np.zeros((n_graphs * len(node_names), self.n_times), dtype=np.float32)
            for idx in range(n_graphs):
                for i in range(len(node_names)):
                    node_features[idx*len(node_names)+i] = np.array([
                        item["data"][x][idx*self.n_times:(idx+1)*self.n_times]
                        for x, name in enumerate(item["ch_names"])
                        if node_names[i] in name
                    ]).mean(axis=0)

        elif self.node_transform == "bipolar":
            node_names = [name.replace("EEG ", "") for name in item["ch_names"]]
            node_positions = np.vstack([
                np.expand_dims(item["sources"], 0),
                np.expand_dims(item["targets"], 0)
            ]).mean(axis=0)

            node_features = np.zeros((n_graphs * len(node_names), self.n_times), dtype=np.float32)
            for idx in range(n_graphs):
                for i in range(len(node_names)):
                    node_features[idx*len(node_names)+i] = item["data"][i][idx*self.n_times:(idx+1)*self.n_times]

        adjecancy = 0
        if "norm_" in self.edge_select:
            pass
        if "static_" in self.edge_select:
            pass
        if "dynamic_" in self.edge_select:
            with np.errstate(divide="ignore", invalid="ignore"):
                adjecancy = np.corrcoef(node_features)

        # Edge Select
        node_count = len(node_names)
        adjecancy_matrix = np.zeros((n_graphs * node_count, n_graphs * node_count), dtype=np.float64)
        for idx in range(n_graphs):
            for i in range(node_count):
                # Cross graph connections (before,after)
                for c in range(self.cross_connections):
                    if idx + c <= n_graphs:
                        adjecancy_matrix[(idx*node_count)+i, ((idx+c+1)*node_count)+i] = 1

                # Inter graph connections (far/close/static/dynamic)
                for j in range(node_count):
                    if i == j:
                        continue

                    value = 0
                    if "norm_" in self.edge_select:
                        value = np.linalg.norm(node_positions[j] - node_positions[i])
                    if "static_" in self.edge_select:
                        distance = self.distances.loc[(self.distances["from"] == f"EEG {node_names[i]}") & (self.distances["to"] == f"EEG {node_names[j]}")]
                        if len(distance) > 0:
                            value = distance.iloc[0]["distance"]
                    if "dynamic_" in self.edge_select:
                        value = adjecancy[i, j]

                    if "_gt" in self.edge_select and value > self.threshold:
                        adjecancy_matrix[idx*node_count+i, idx*node_count+j] = 1
                    if "_lt" in self.edge_select and value < self.threshold:
                        adjecancy_matrix[idx*node_count+i, idx*node_count+j] = 1

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(item["labels"].max()),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
            n_nodes=pt.tensor(node_count),
            n_graphs=pt.tensor(n_graphs),
            node_positions=pt.from_numpy(node_positions),
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Brain2Vec")
        parser.add_argument("--n_times", type=int, default=256)
        parser.add_argument("--n_outputs", type=int, default=2)
        parser.add_argument("--loss_fn", type=str, default="ce", choices=["ce", "focal"])

        parser.add_argument("--cross_connections", type=int, default=1)
        parser.add_argument("--signal_transform", type=str, default="raw", choices=["raw", "fourier", "wavelet"])
        parser.add_argument("--node_transform", type=str, default="unipolar", choices=["unipolar", "bipolar"])
        parser.add_argument("--edge_select", type=str, default="norm_lt", choices=["norm_lt", "norm_gt", "static_lt", "static_gt", "dynamic_lt", "dynamic_gt"])
        parser.add_argument("--threshold", type=float, default=0.1)

        parser.add_argument("--aggregator", type=str, default="sequence", choices=["sequence", "vector"])
        parser.add_argument("--gru_size", type=int, default=4)

        return parent_parser
