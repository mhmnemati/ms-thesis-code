import torch as T
import lightning as L
import torchmetrics as M
import torch.nn.functional as F

from torch_geometric.data.batch import Batch


class BaseModel(L.LightningModule):
    def __init__(self, get_model, num_classes):
        super().__init__()
        self.model = get_model()
        self.training_metrics = {
            "f1": M.F1Score(task="multiclass", num_classes=num_classes),
            "recall": M.Recall(task="multiclass", num_classes=num_classes),
            "precision": M.Precision(task="multiclass", num_classes=num_classes),
            "accuracy": M.Accuracy(task="multiclass", num_classes=num_classes)
        }
        self.validation_metrics = {
            "f1": M.F1Score(task="multiclass", num_classes=num_classes),
            "recall": M.Recall(task="multiclass", num_classes=num_classes),
            "precision": M.Precision(task="multiclass", num_classes=num_classes),
            "accuracy": M.Accuracy(task="multiclass", num_classes=num_classes)
        }

    def forward(self, *args):
        return self.model(*args)

    def configure_optimizers(self):
        return T.optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, idx):
        batch_size, args, y = 0, 0, 0
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            args = (batch.x, batch.edge_index, batch.batch)
            y = batch.y
        else:
            batch_size = len(batch[1])
            args = (batch[0],)
            y = batch[1]

        pred = self.model(*args)
        loss = F.cross_entropy(pred, y)

        self.log("training_loss", loss, batch_size=batch_size)
        for key, val in self.training_metrics.items():
            self.log(f"training_{key}", val(pred, y), batch_size=batch_size)

        return loss

    def validation_step(self, batch, idx):
        batch_size, args, y = 0, 0, 0
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            args = (batch.x, batch.edge_index, batch.batch)
            y = batch.y
        else:
            batch_size = len(batch[1])
            args = (batch[0],)
            y = batch[1]

        pred = self.model(*args)
        loss = F.cross_entropy(pred, y)

        self.log("validation_loss", loss, batch_size=batch_size)
        for key, val in self.validation_metrics.items():
            self.log(f"validation_{key}", val(pred, y), batch_size=batch_size)
