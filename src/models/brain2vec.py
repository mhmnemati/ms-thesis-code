import torch.nn as T
import torch_geometric.nn as G
import torch.nn.functional as F

from .base import BaseModel


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

        self.model = G.Sequential("x, edge_index, graph_size, graph_length, batch", [
            (Conv(in_channels=n_times, out_channels=int(n_times/2)), "x, edge_index -> x"),
            (T.ReLU(), "x -> x"),
            (Conv(in_channels=int(n_times/2), out_channels=int(n_times/4)), "x, edge_index -> x"),
            (Agg(), "x, batch -> x"),
            (T.Dropout(p=0.5), "x -> x"),
            (T.Linear(in_features=int(n_times/4), out_features=n_outputs), "x -> x"),
            (T.LogSoftmax(dim=1), "x -> x")
        ])

    def forward(self, x, edge_index, graph_size, graph_length, batch):
        return self.model(x, edge_index, graph_size, graph_length, batch)


class Brain2Vec(BaseModel):
    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs", "layer_type", "aggregator"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=F.cross_entropy
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Brain2Vec")
        parser.add_argument("--n_times", type=int, default=3000)
        parser.add_argument("--n_outputs", type=int, default=2)
        parser.add_argument("--layer_type", type=str, default="gcn", choices=["gcn", "gcn2", "gat", "gat2", "cheb"])
        parser.add_argument("--aggregator", type=str, default="min", choices=["min", "max", "mean", "median"])
        return parent_parser
