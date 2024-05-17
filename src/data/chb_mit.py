import os
import random
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
    electrode_distances = pd.read_csv(f"{os.path.dirname(__file__)}/distances_3d.csv")

    def __init__(self, fold, folds, **kwargs):
        transform = self.tensor2vec
        data_loader = TensorDataLoader
        if kwargs["batch_type"] == "graph2vec":
            transform = self.graph2vec
            data_loader = GraphDataLoader
        elif kwargs["batch_type"] == "graph2seq":
            transform = self.graph2seq
            data_loader = GraphDataLoader

        self.edge_select = kwargs["edge_select"]
        self.wave_transform = kwargs["wave_transform"]

        super().__init__(
            name="chb_mit_window_1",
            filters=self.filters(fold, folds),
            transform=transform,
            data_loader=data_loader,
            batch_size=kwargs["batch_size"],
        )

    def filters(self, fold, folds):
        def kfold(items, train):
            np.random.seed(self.seed)
            np.random.shuffle(items)

            patients = np.arange(1, 25)
            np.random.seed(self.seed)
            np.random.shuffle(patients)
            parts = np.array_split(patients, folds)

            valid_patients = parts[fold]
            train_patients = np.array([p for p in patients if p not in valid_patients])

            if folds == 1:
                temp = train_patients
                train_patients = valid_patients
                valid_patients = temp

            def my_filter(x):
                patients = valid_patients
                if train:
                    patients = train_patients

                for i in patients:
                    if f"chb{i:02d}" in x:
                        return True

                return False

            return filter(my_filter, items)

        return {
            "train": lambda items: list(kfold(items, True)),
            "valid": lambda items: list(kfold(items, False)),
        }

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

        electrodes = list(set(sources + targets))
        n_electrodes = len(electrodes)
        electrode2id = {val: idx for idx, val in enumerate(electrodes)}
        id2electrode = {val: idx for idx, val in electrode2id.items()}

        n_graphs = 1 if (full == True) else labels.shape[0]
        n_times = int(data.shape[1] / n_graphs)

        node_features = np.zeros((n_electrodes, n_times), dtype=np.float32)
        for i in range(data.shape[0]):
            # Convert bipolar wave data to electrode node_features
            if self.wave_transform == "power":
                power = data[i, :] ** 2
                source_idx = electrode2id[sources[i]]
                target_idx = electrode2id[targets[i]]
                node_features[source_idx] += power / 2
                node_features[target_idx] += power / 2
            elif self.wave_transform == "fourier":
                # TODO: implementation needed
                pass
            elif self.wave_transform == "wavelet":
                # TODO: implementation needed
                pass

        adjecancy_matrix = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)
        for i in range(n_electrodes):
            # Inter graph connections (const/cluster/dynamic/...)
            for j in range(n_electrodes):
                data = self.electrode_distances
                distance = data.loc[(data["from"] == f"EEG {id2electrode[i]}") & (data["to"] == f"EEG {id2electrode[j]}")]

                if len(distance) > 0:
                    distance = distance.iloc[0]["distance"]
                    if distance > 0.9:
                        adjecancy_matrix[i, j] = 1

                # if self.edge_select == "far":
                #     distance = np.linalg.norm(id2electrode[j] - id2electrode[i])
                #     adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance > 0.1 else 0
                # elif self.edge_select == "close":
                #     distance = np.linalg.norm(electrodes[j] - electrodes[i])
                #     adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance < 0.1 else 0
                # elif self.edge_select == "cluster":
                #     # TODO: implementation needed
                #     pass
                # elif self.edge_select == "dynamic":
                #     # TODO: implementation needed
                #     pass

        return Data(
            x=pt.from_numpy(node_features),
            y=pt.tensor(labels.max() if (full == True) else labels.reshape(1, -1)),
            edge_index=from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0],
            graph_size=pt.tensor(n_electrodes),
            graph_length=pt.tensor(n_graphs),
        )

    def graph2vec(self, item):
        return self.get_graph(True, item["data"], item["labels"], item["sources"], item["targets"])

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--batch_type", type=str, default="tensor2vec", choices=["tensor2vec", "graph2vec"])
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="power", choices=["power", "fourier", "wavelet"])
        return parent_parser
