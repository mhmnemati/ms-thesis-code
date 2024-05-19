import torch
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from models import Deep4Net, EEGInception, Brain2Vec, Brain2Seq, GCNBiGRU
from data import CHBMIT, SleepEDFX

torch.multiprocessing.set_sharing_strategy("file_system")

model_classes = {
    "deep4net": Deep4Net,
    "eeginception": EEGInception,
    "brain2vec": Brain2Vec,
    "brain2seq": Brain2Seq,
    "gcnbigru": GCNBiGRU,
}
data_classes = {
    "chb_mit": CHBMIT,
    "sleep_edfx": SleepEDFX,
}

parser = argparse.ArgumentParser(description="Train model on data.")
parser.add_argument("-v", "--version", type=str, default=None)
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-m", "--model", type=str, default=list(model_classes.keys())[0], choices=list(model_classes.keys()))
parser.add_argument("-d", "--data", type=str, default=list(data_classes.keys())[0], choices=list(data_classes.keys()))

temp_args, _ = parser.parse_known_args()
Model = model_classes[temp_args.model]
Data = data_classes[temp_args.data]
parser = Model.add_arguments(parser)
parser = Data.add_arguments(parser)

args = parser.parse_args()

trainer = L.Trainer(max_epochs=args.epochs, logger=TensorBoardLogger("logs/", name=f"{args.model}/{args.data}", version=args.version, default_hp_metric=False))
trainer.fit(Model(**vars(args)), datamodule=Data(**vars(args)))
