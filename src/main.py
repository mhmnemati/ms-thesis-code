import lightning as L

from data import CHBMITDataset
from torch.utils.data import DataLoader
from models import EEGInception


def transform(item):
    return (item["data"], item["label"])


BATCH = 8

train_set = CHBMITDataset(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=4)

test_set = CHBMITDataset(train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH, num_workers=4)

model = EEGInception()
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, train_loader, test_loader)
