import os
import pywt
import numpy as np
import scipy as sp
import torch as pt
import pandas as pd

from torch_geometric.data import Data
from torch.utils.data import DataLoader as TensorDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseDataset


class CHBMIT(BaseDataset):
    seed = 100
    distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, **kwargs):
        transform = self.tensor2vec
        data_loader = TensorDataLoader
        if kwargs["batch_type"] == "graph2vec":
            transform = self.graph2vec
            data_loader = GraphDataLoader
        elif kwargs["batch_type"] == "graph2seq":
            transform = self.graph2seq
            data_loader = GraphDataLoader

        self.k = kwargs["k"]
        self.folds = kwargs["folds"]
        self.edge_select = kwargs["edge_select"]
        self.wave_transform = kwargs["wave_transform"]

        super().__init__(
            name="chb_mit_window_1",
            filters=self.filters,
            transform=transform,
            data_loader=data_loader,
            num_workers=kwargs["num_workers"],
            batch_size=kwargs["batch_size"],
        )

    def filters(self, stage):
        def select(items):
            np.random.seed(self.seed)
            np.random.shuffle(items)

            if stage in ["test", "predict"]:
                return items

            all_patients = np.arange(1, 25)
            np.random.seed(self.seed)
            np.random.shuffle(all_patients)
            parts = np.array_split(all_patients, self.folds)

            patients = np.concatenate(parts[:self.k] + parts[self.k+1:]) if stage == "train" else parts[self.k]
            patients = [f"chb{p:02d}" for p in patients]

            return list(filter(lambda item: any([p in item for p in patients]), items))

        return select

    def tensor2vec(self, item):
        return (item["data"], item["labels"].max())

    def get_graph(self, full, data, labels, sources, targets, ch_names):
        source_names = [name.replace("EEG ", "").split("-")[0] for name in ch_names]
        target_names = [name.replace("EEG ", "").split("-")[1] for name in ch_names]

        electrode_positions = np.concatenate([sources, targets])
        electrode_names = (source_names + target_names)
        electrodes = list(set(electrode_names))
        n_electrodes = len(electrodes)
        n_graphs = 1 if (full == True) else labels.shape[0]
        n_times = int(data.shape[1] / n_graphs)

        node_features = np.zeros((n_electrodes * n_graphs, n_times), dtype=np.float32)
        for idx in range(n_graphs):
            for i in range(data.shape[0]):
                # Convert bipolar wave data to electrode node_features
                power = data[i, idx*n_times:(idx+1)*n_times]

                if self.wave_transform == "power":
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
            y=pt.tensor(labels.max() if (full == True) else labels.reshape(1, -1)),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
            graph_size=pt.tensor(n_electrodes),
            graph_length=pt.tensor(n_graphs),
        )

    def graph2vec(self, item):
        return self.get_graph(True, item["data"], item["labels"], item["sources"], item["targets"], item["ch_names"])

    def graph2seq(self, item):
        return self.get_graph(False, item["data"], item["labels"], item["sources"], item["targets"], item["ch_names"])

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--k", type=int, default=1)
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--batch_type", type=str, default="tensor2vec", choices=["tensor2vec", "graph2vec", "graph2seq"])
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="power", choices=["power", "fourier", "wavelet"])
        return parent_parser
