from .base import BaseModel
import torch.nn.functional as F
import braindecode.models as M


class DeepSleepNet(BaseModel):
    def __init__(self):
        super().__init__(
            num_classes=2,
            model=M.DeepSleepNet(n_times=3000, n_chans=23, n_outputs=2),
            loss=F.cross_entropy,
        )

    @staticmethod
    def transform(item):
        return (item["data"], item["label"])
