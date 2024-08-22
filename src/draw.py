# import mne
# import wfdb
# import matplotlib

# matplotlib.use("webagg")

# record = "/root/pytorch_datasets/chb_mit_raw/chb15/chb15_06.edf"
# raw = mne.io.read_raw_edf(record, infer_types=True, include=[
#     "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
#     "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
#     "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
# ])

# seizure = wfdb.io.rdann(record, extension="seizures")
# start = seizure.sample[0] / raw.info["sfreq"]
# finish = seizure.sample[1] / raw.info["sfreq"]
# raw.set_annotations(mne.Annotations(onset=start, duration=(finish - start), description="seizure"))

# data = raw.get_data() * 1e6

# seizure = data[:, int(raw.info["sfreq"] * 272):int(raw.info["sfreq"] * 397)]
# print(seizure.min())
# print(seizure.mean())
# print(seizure.max())

# for i in range(1, 6):
#     n_seizure = data[:, int(raw.info["sfreq"] * (272 + (i * 100))):int(raw.info["sfreq"] * (397 + (i * 100)))]

#     print("----------------------------")
#     print(n_seizure.min())
#     print(n_seizure.mean())
#     print(n_seizure.max())

# raw.plot()

# input()

import torch
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models import Brain2Vec
from data import TensorDataset
from torchviz import make_dot

model = Brain2Vec(
    n_times=256,
    n_outputs=2,
    loss_fn="ce",
    normalization="micro",
    cross_connections=1,
    signal_transform="raw",
    node_transform="unipolar",
    edge_select="norm_lt",
    threshold=0.1,
    aggregator="vector",
    gru_size=4,
)

dataset = TensorDataset(
    name="chb_mit_window_5",
    split="train",
    method="kfold",
    folds=5,
    k=-1,
    transform=model.transform,
)

x = dataset[5]
y = model(x.x, x.edge_index, x.n_nodes, x.n_graphs, x.batch)
params = dict(model.model.named_parameters())
make_dot(y.mean(), params=params)
# input()
