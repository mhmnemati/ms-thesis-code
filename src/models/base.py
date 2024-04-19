import torch as T
import lightning as L
import torchmetrics as M
import torch.nn.functional as F

from torch_geometric.data.batch import Batch


class BaseModel(L.LightningModule):
    def __init__(self, get_model, num_classes):
        super().__init__()
        self.model = get_model()

        self.training_f1 = M.F1Score(task="multiclass", num_classes=num_classes)
        self.training_recall = M.Recall(task="multiclass", num_classes=num_classes)
        self.training_precision = M.Precision(task="multiclass", num_classes=num_classes)
        self.training_accuracy = M.Accuracy(task="multiclass", num_classes=num_classes)

        self.validation_f1 = M.F1Score(task="multiclass", num_classes=num_classes)
        self.validation_recall = M.Recall(task="multiclass", num_classes=num_classes)
        self.validation_precision = M.Precision(task="multiclass", num_classes=num_classes)
        self.validation_accuracy = M.Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log("training_f1", self.training_f1(pred, y), batch_size=batch_size)
        self.log("training_recall", self.training_recall(pred, y), batch_size=batch_size)
        self.log("training_precision", self.training_precision(pred, y), batch_size=batch_size)
        self.log("training_accuracy", self.training_accuracy(pred, y), batch_size=batch_size)
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
        self.log("validation_f1", self.validation_f1(pred, y), batch_size=batch_size)
        self.log("validation_recall", self.validation_recall(pred, y), batch_size=batch_size)
        self.log("validation_precision", self.validation_precision(pred, y), batch_size=batch_size)
        self.log("validation_accuracy", self.validation_accuracy(pred, y), batch_size=batch_size)
