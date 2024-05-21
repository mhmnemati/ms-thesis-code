import numpy as np
import scipy as sp
import torch as pt

from torch_geometric.data import Data
from torch.utils.data import DataLoader as TensorDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseDataset


class SleepEDFX(BaseDataset):
    seed = 100

    def __init__(self, **kwargs):
        transform = self.tensor2vec
        data_loader = TensorDataLoader
        if kwargs["batch_type"] == "graph2vec":
            transform = self.graph2vec
            data_loader = GraphDataLoader
        elif kwargs["batch_type"] == "graph2seq":
            transform = self.graph2seq
            data_loader = GraphDataLoader
        elif kwargs["batch_type"] == "biot":
            transform = self.biot
            data_loader = TensorDataLoader

        self.k = kwargs["k"]
        self.folds = kwargs["folds"]
        self.edge_select = kwargs["edge_select"]
        self.wave_transform = kwargs["wave_transform"]

        super().__init__(
            name="sleep_edfx_window_30_overlap_0",
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

            all_patients = np.array(list(set([item.split("/")[-1][:5] for item in items])))
            np.random.seed(self.seed)
            np.random.shuffle(all_patients)
            parts = np.array_split(all_patients, self.folds)

            patients = np.concatenate(parts[:self.k] + parts[self.k+1:]) if stage == "train" else parts[self.k]
            patients = [f"{p}" for p in patients]

            return list(filter(lambda item: any([p in item for p in patients]), items))

        return select

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
                # Convert bipolar wave data to electrode node_features
                if self.wave_transform == "power":
                    power = data[i, idx*n_times:(idx+1)*n_times] ** 2
                    source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
                    target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()
                    node_features[source_idx] += power / 2
                    node_features[target_idx] += power / 2
                elif self.wave_transform == "fourier":
                    # TODO: implementation needed
                    pass
                elif self.wave_transform == "wavelet":
                    # TODO: implementation needed
                    pass

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
                        distance = np.linalg.norm(electrodes[j] - electrodes[i])
                        adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance > 0.1 else 0
                    elif self.edge_select == "close":
                        distance = np.linalg.norm(electrodes[j] - electrodes[i])
                        adjecancy_matrix[(idx*n_electrodes)+i, (idx*n_electrodes)+j] = 1 if distance < 0.1 else 0
                    elif self.edge_select == "cluster":
                        # TODO: implementation needed
                        pass
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
        return self.get_graph(True, item["data"], item["labels"], item["sources"], item["targets"])

    def graph2seq(self, item):
        return self.get_graph(False, item["data"], item["labels"], item["sources"], item["targets"])

    def biot(self, item):
        channels = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "C3-A2",
            "C4-A1",
        ]

        data = np.zeros((len(channels), 30 * 200), dtype=np.float32)

        for idx, ch_name in enumerate(item["ch_names"]):
            if "EEG" not in ch_name:
                continue

            if ch_name.replace("EEG ", "") == "Fpz-Cz":
                signal = sp.signal.resample(item["data"][idx], 30 * 200) / 2
                data[channels.index("FP1-F3")] = signal
                data[channels.index("F3-C3")] = signal
                data[channels.index("FP2-F4")] = signal
                data[channels.index("F4-C4")] = signal

            if ch_name.replace("EEG ", "") == "Pz-Oz":
                signal = sp.signal.resample(item["data"][idx], 30 * 200)
                data[channels.index("P3-O1")] = signal
                data[channels.index("P4-O2")] = signal

        return (data, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--k", type=int, default=1)
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--batch_type", type=str, default="tensor2vec", choices=["tensor2vec", "graph2vec", "graph2seq", "biot"])
        parser.add_argument("--edge_select", type=str, default="far", choices=["far", "close", "cluster", "dynamic"])
        parser.add_argument("--wave_transform", type=str, default="power", choices=["power", "fourier", "wavelet"])
        return parent_parser
