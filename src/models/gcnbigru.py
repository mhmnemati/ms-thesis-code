import torch.nn as T
import torch_geometric.nn as G
import torch.nn.functional as F

from .base import BaseModel


class Model(T.Module):
    def __init__(self, n_times, n_outputs):
        super().__init__()

        self.model = G.Sequential("x, edge_index, graph_size, graph_length, batch", [
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
            (T.LogSoftmax(dim=1), "x -> x")
        ])

    def forward(self, *args):
        return self.model(*args)


class GCNBiGRU(BaseModel):
    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=F.cross_entropy
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("GCNBiGRU")
        parser.add_argument("--n_times", type=int, default=256)
        parser.add_argument("--n_outputs", type=int, default=2)
        return parent_parser
