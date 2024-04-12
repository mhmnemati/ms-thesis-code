import torch as T
import lightning as L

from data import CHBMITDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import Brain2Vec

from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix


import numpy as np


def transform(item):
    data = item["data"]
    sources = item["sources"]
    targets = item["targets"]

    electrodes = np.unique(np.concatenate([sources, targets]), axis=0)

    node_features = np.zeros((electrodes.shape[0], data.shape[1]), dtype=np.float32)
    for i in range(len(data)):
        power = data[i] ** 2

        source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
        target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

        node_features[source_idx] += power / 2
        node_features[target_idx] += power / 2

    adjecancy_matrix = np.zeros((electrodes.shape[0], electrodes.shape[0]), dtype=np.float64)
    for i in range(electrodes.shape[0]):
        for j in range(electrodes.shape[0]):
            distance = np.linalg.norm(electrodes[j] - electrodes[i])
            adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0

    edge_index = from_scipy_sparse_matrix(sparse.csr_matrix(adjecancy_matrix))[0]

    return Data(x=T.from_numpy(node_features), edge_index=edge_index, y=T.tensor([item["label"]], dtype=T.int64))


BATCH = 8

train_set = CHBMITDataset(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=4)

test_set = CHBMITDataset(train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH, num_workers=4)

model = Brain2Vec()
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, train_loader, test_loader)
