import warnings
import torch as pt
import lightning as L
import torchmetrics as M
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from torch_geometric.data.batch import Batch


class BaseModel(L.LightningModule):
    def __init__(self, num_classes, hparams, model, loss):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.loss = loss

        if num_classes > 1:
            self.confusion_matrix = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)
            self.metrics = M.MetricCollection({
                "f1": M.F1Score(task="multiclass", num_classes=num_classes),
                "recall": M.Recall(task="multiclass", num_classes=num_classes),
                "accuracy": M.Accuracy(task="multiclass", num_classes=num_classes),
                "precision": M.Precision(task="multiclass", num_classes=num_classes),
            })
        else:
            self.confusion_matrix = M.ConfusionMatrix(task="binary")
            self.metrics = M.MetricCollection({
                "f1": M.F1Score(task="binary"),
                "recall": M.Recall(task="binary"),
                "accuracy": M.Accuracy(task="binary"),
                "precision": M.Precision(task="binary"),
            })

        self.training_metrics = self.metrics.clone(prefix="training/")
        self.validation_metrics = self.metrics.clone(prefix="validation/")

        self.test_interpreter = IntegratedGradients(model)

    def forward(self, *args):
        return self.model(*args)

    def configure_optimizers(self):
        optimizer = pt.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = pt.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            "training/f1": 0,
            "training/recall": 0,
            "training/accuracy": 0,
            "training/precision": 0,
        })

    def general_step(self, batch):
        batch_size, args, y = 0, 0, 0
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            args = (batch.x, batch.edge_index, batch.batch)
            y = batch.y

            if "n_nodes" in batch:
                args = (batch.x, batch.edge_index, batch.n_nodes, batch.n_graphs, batch.batch)
        else:
            batch_size = len(batch[-1])
            args = batch[:-1]
            y = batch[-1]

        pred = self.model(*args)

        if pred.shape[-1] == 1:
            pred = pred.view(-1)
            y = y.float()

        loss = self.loss(pred, y)

        if len(pred.shape) > 1:
            pred = pred.argmax(-1)

        return (batch_size, args, loss, pred, y)

    def training_step(self, batch):
        batch_size, _, loss, pred, y = self.general_step(batch)

        self.log("training/loss", loss, batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log_dict(self.training_metrics(pred, y), batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch):
        batch_size, _, loss, pred, y = self.general_step(batch)

        self.log("validation/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log_dict(self.validation_metrics(pred, y), batch_size=batch_size, sync_dist=True)

        self.confusion_matrix.update(pred, y)

    def test_step(self, batch):
        batch_size, args, _, pred, y = self.general_step(batch)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            attrs, = self.interpreter.attribute(internal_batch_size=batch_size, inputs=args[:1], additional_forward_args=args[1:], target=y)
            print(pt.mean(attrs, dim=-1))

    def on_validation_epoch_end(self):
        fig, _ = self.confusion_matrix.plot()
        self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
        self.confusion_matrix.reset()
        plt.close(fig)
