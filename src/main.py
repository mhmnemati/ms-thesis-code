import torch
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from models import Deep4Net, EEGInception, Brain2Vec, Brain2Seq, GCNBiGRU, GCNAttn, BIOT
from data import TensorDataset

torch.multiprocessing.set_sharing_strategy("file_system")

model_classes = {
    "deep4net": Deep4Net,
    "eeginception": EEGInception,
    "brain2vec": Brain2Vec,
    "brain2seq": Brain2Seq,
    "gcnbigru": GCNBiGRU,
    "gcnattn": GCNAttn,
    "biot": BIOT,
}
data_sets = [
    "chb_mit_window_1",
    "chb_mit_window_5",
    "chb_mit_window_15",
    "chb_mit_window_30",
    "sleep_edfx_window_30_overlap_0"
]

parser = argparse.ArgumentParser(description="Train model on data.")

parser.add_argument("-v", "--version", type=str, default=None)
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-m", "--model", type=str, default=list(model_classes.keys())[0], choices=list(model_classes.keys()))
parser.add_argument("-d", "--data", type=str, default=data_sets[0], choices=data_sets)
parser.add_argument("-n", "--num_workers", type=int, default=2)
parser.add_argument("-b", "--batch_size", type=int, default=8)
parser.add_argument("-f", "--folds", type=int, default=5)
parser.add_argument("-k", "--k", type=int, default=1)

temp_args, _ = parser.parse_known_args()
Model = model_classes[temp_args.model]
parser = Model.add_arguments(parser)
args = parser.parse_args()

model = Model(**vars(args))

train_set = TensorDataset(name=args.data, split="train", folds=args.folds, k=-(args.k), transform=model.transform)
valid_set = TensorDataset(name=args.data, split="train", folds=args.folds, k=+(args.k), transform=model.transform)
train_loader = model.data_loader(train_set, num_workers=args.num_workers, batch_size=args.batch_size)
valid_loader = model.data_loader(valid_set, num_workers=args.num_workers, batch_size=args.batch_size)

trainer = L.Trainer(max_epochs=args.epochs, logger=TensorBoardLogger("logs/", name=f"{args.model}/{args.data}", version=args.version, default_hp_metric=False))
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
