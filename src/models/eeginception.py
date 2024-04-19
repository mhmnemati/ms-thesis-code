from .base import BaseModel
import torch.nn.functional as F
import braindecode.models as M


class EEGInception(BaseModel):
    def __init__(self):
        super().__init__(
            get_model=lambda: M.EEGInception(n_times=3000, n_chans=23, n_outputs=2),
            get_loss=lambda: F.cross_entropy,
            num_classes=2
        )

    @staticmethod
    def transform(item):
        return (item["data"], item["label"])
