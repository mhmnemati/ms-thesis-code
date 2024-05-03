import torch as T
import lightning as L
import torchmetrics as M
import torch.nn.functional as F

from torch_geometric.data.batch import Batch


class BaseModel(L.LightningModule):
    def __init__(self, num_classes, hparams, model, loss):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.num_classes = num_classes
        self.model = model
        self.loss = loss

        metrics = M.MetricCollection({
            "f1": M.F1Score(task="multiclass", num_classes=num_classes),
            "aucroc": M.AUROC(task="multiclass", num_classes=num_classes),
            "recall": M.Recall(task="multiclass", num_classes=num_classes),
            "accuracy": M.Accuracy(task="multiclass", num_classes=num_classes),
            "precision": M.Precision(task="multiclass", num_classes=num_classes),
        })

        self.training_metrics = metrics.clone(prefix="training/")
        self.validation_metrics = metrics.clone(prefix="validation/")
        self.validation_matrix = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, *args):
        return self.model(*args)

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            "training/f1": 0,
            "training/aucroc": 0,
            "training/accuracy": 0,
        })

    def general_step(self, batch):
        batch_size, args, y = 0, 0, 0
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            args = (batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(-1)
        else:
            batch_size = len(batch[1])
            args = (batch[0],)
            y = batch[1].view(-1)

        pred = self.model(*args)
        loss = self.loss(pred, y)

        return (batch_size, loss, pred, y)

    def training_step(self, batch, idx):
        batch_size, loss, pred, y = self.general_step(batch)

        self.log("training/loss", loss, batch_size=batch_size, prog_bar=True)
        self.log_dict(self.training_metrics(pred, y), batch_size=batch_size, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):
        batch_size, loss, pred, y = self.general_step(batch)

        self.log("validation/loss", loss, batch_size=batch_size)
        self.log_dict(self.validation_metrics(pred, y), batch_size=batch_size)

        self.validation_matrix.update(pred, y)

    def on_validation_epoch_end(self):
        fig, _ = self.validation_matrix.plot()
        self.logger.experiment.add_figure("validation_matrix", fig, self.current_epoch)
        self.validation_matrix.reset()
