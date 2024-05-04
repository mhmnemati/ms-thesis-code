import torch as pt
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

            # batch = [0[3*5], 1[2*4], 2[1*3]]
            # batch_size = 3
            # graph_size = [3,2,1]
            # graph_length = [5,4,3]
            # batch_old = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2] = (26)
            # batch_new = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4, 5,5,6,6,7,7,8,8, 9,10,11] = (26)
            # Caution: this implementation is highly optimized and complex
            (lambda graph_size, graph_length: pt.repeat_interleave(pt.arange(graph_length.sum()), pt.repeat_interleave(graph_size, graph_length)), "graph_size, graph_length -> batch"),

            (Agg(), "x, batch -> x"),
            (T.Dropout(p=0.1), "x -> x"),
            (T.Linear(in_features=int(n_times/4), out_features=n_outputs), "x -> x"),

            # x_old = (12, 2)
            # x_tmp = [(5,2), (4,2), (3,2)]
            # x_new = (3, 5=max(graph_length), 2)
            (lambda x, graph_length: pt.nn.utils.rnn.pad_sequence(x.split(list(graph_length)), batch_first=True), "x, graph_length -> x"),
        ])

    def forward(self, *args):
        return self.model(*args)


class Brain2Seq(BaseModel):
    def __init__(self, **kwargs):
        def loss_fn(pred, true):
            return F.cross_entropy(pred.view(-1, hparams["n_outputs"]), true.view(-1))

        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_outputs", "layer_type", "aggregator"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=Model(**hparams),
            loss=loss_fn
        )

    @ staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Brain2Seq")
        parser.add_argument("--n_times", type=int, default=100)
        parser.add_argument("--n_outputs", type=int, default=2)
        parser.add_argument("--layer_type", type=str, default="gcn", choices=["gcn", "gcn2", "gat", "gat2", "cheb"])
        parser.add_argument("--aggregator", type=str, default="min", choices=["min", "max", "mean", "median"])
        return parent_parser
