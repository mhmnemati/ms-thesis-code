import argparse
import lightning as L
from torch.utils.data import DataLoader as TensorDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

from data import CHBMITDataset
from models import Brain2Vec, EEGInception

data_classes = {
    "chb_mit": CHBMITDataset,
}
model_classes = {
    "brain2vec": (Brain2Vec, GraphDataLoader),
    "eeginception": (EEGInception, TensorDataLoader),
}

parser = argparse.ArgumentParser(description="Train model on data.")
parser.add_argument("-d", "--data", type=str, default=list(data_classes.keys())[0], choices=list(data_classes.keys()))
parser.add_argument("-m", "--model", type=str, default=list(model_classes.keys())[0], choices=list(model_classes.keys()))
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-b", "--batches", type=int, default=8)

args = parser.parse_args()

Dataset = data_classes[args.data]
Model, DataLoader = model_classes[args.model]

train_dataset = Dataset(train=True, transform=Model.transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=int(args.batch/2))

test_dataset = Dataset(train=False, transform=Model.transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=int(args.batch/2))

trainer = L.Trainer(max_epochs=args.epochs)
trainer.fit(Model(), train_loader, test_loader)
