import torch as T
import lightning as L
import torchmetrics as M
import torch.nn.functional as F
from braindecode.models import EEGInception as Model


class EEGInception(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model(n_outputs=2, n_chans=23, n_times=3000)
        self.training_accuracy = M.Accuracy(task="multiclass", num_classes=2)
        self.validation_accuracy = M.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return T.optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, idx):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        self.log("training_loss", loss)
        self.log("training_accuracy", self.training_accuracy(pred, y))
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        self.log("validation_loss", loss)
        self.log("validation_accuracy", self.validation_accuracy(pred, y))


def preprocess(item):
    return (item["data"], item["label"])
