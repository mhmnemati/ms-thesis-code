from .base import BaseModel
import torch as pt
import torch.nn.functional as F
import braindecode.models as M


class Deep4Net(BaseModel):
    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_times", "n_chans", "n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=M.Deep4Net(**hparams),
            loss=pt.nn.CrossEntropyLoss(weight=pt.tensor([0.01, 10000]))
        )

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Deep4Net")
        parser.add_argument("--n_times", type=int, default=3000)
        parser.add_argument("--n_chans", type=int, default=23)
        parser.add_argument("--n_outputs", type=int, default=2)
        return parent_parser
