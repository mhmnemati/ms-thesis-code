from .base import BaseModel
from braindecode.models import EEGInception as Model


class EEGInception(BaseModel):
    def __init__(self):
        super().__init__(
            get_model=lambda: Model(n_outputs=2, n_chans=23, n_times=3000),
            num_classes=2
        )

    @staticmethod
    def transform(item):
        return (item["data"], item["label"])
