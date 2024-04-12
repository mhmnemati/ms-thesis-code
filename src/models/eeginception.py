from .base import BaseModel
from torch.utils.data import DataLoader
from braindecode.models import EEGInception as Model


def transform(item):
    return (item["data"], item["label"])


class EEGInception(BaseModel):
    def __init__(self):
        super().__init__(
            get_model=lambda: Model(n_outputs=2, n_chans=23, n_times=3000),
            num_classes=2
        )

    @staticmethod
    def loader(Dataset, **kwargs):
        dataset = Dataset(**kwargs, transform=transform)
        return DataLoader(dataset, batch_size=8, num_workers=4)
