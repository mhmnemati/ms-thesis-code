import torch
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from models import Brain2Vec, EEGInception
from data import CHBMITDataset

torch.multiprocessing.set_sharing_strategy("file_system")

model_classes = {
    "brain2vec": Brain2Vec,
    "eeginception": EEGInception,
}
data_classes = {
    "chb_mit": CHBMITDataset,
}

parser = argparse.ArgumentParser(description="Train model on data.")
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-m", "--model", type=str, default=list(model_classes.keys())[0], choices=list(model_classes.keys()))
parser.add_argument("-d", "--data", type=str, default=list(data_classes.keys())[0], choices=list(data_classes.keys()))

temp_args, _ = parser.parse_known_args()
Model = model_classes[temp_args.model]
Data = data_classes[temp_args.data]
parser = Model.add_arguments(parser)
parser = Data.add_arguments(parser)

args = parser.parse_args()

trainer = L.Trainer(max_epochs=args.epochs, logger=TensorBoardLogger(save_dir="logs/", name=f"{args.model}/{args.data}", default_hp_metric=False))
trainer.fit(Model(**vars(args)), datamodule=Data(**vars(args)))
