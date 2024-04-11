import lightning as L

from data import CHBMITDataset
from torch.utils.data import DataLoader
from models import EEGInception


import numpy as np


def transform(item):
    data = item["data"]
    sources = item["sources"]
    targets = item["targets"]

    electrodes = np.unique(np.concatenate([sources, targets]), axis=0)

    node_features = np.zeros((electrodes.shape[0], data.shape[1]))
    for i in range(len(data)):
        power = data[i] ** 2

        source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
        target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

        node_features[source_idx] += power / 2
        node_features[target_idx] += power / 2

    adjecancy_matrix = np.zeros((electrodes.shape[0], electrodes.shape[0]))
    for i in range(electrodes.shape[0]):
        for j in range(electrodes.shape[0]):
            distance = np.linalg.norm(electrodes[j] - electrodes[i])
            adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0

    print(node_features)
    print(adjecancy_matrix)

    return (item["data"], item["label"])


BATCH = 8

train_set = CHBMITDataset(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=4)

test_set = CHBMITDataset(train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH, num_workers=4)

train_set[5]

# model = EEGInception()
# trainer = L.Trainer(max_epochs=5)
# trainer.fit(model, train_loader, test_loader)
