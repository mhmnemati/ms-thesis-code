import torch as T
import lightning as L
import torchmetrics as M
import torch.nn.functional as F

import torch.nn as tnn
import torch_geometric.nn as gnn


class Brain2Vec(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels=3000, out_channels=2000)
        self.conv2 = gnn.GCNConv(in_channels=2000, out_channels=1000)
        self.linear = tnn.Linear(in_features=1000, out_features=2)

        self.training_accuracy = M.Accuracy(task="multiclass", num_classes=2)
        self.validation_accuracy = M.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = gnn.global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        x = F.softmax(x)
        return x

    def configure_optimizers(self):
        return T.optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, idx):
        x = self.conv1(batch.x, batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index)
        x = gnn.global_mean_pool(x, batch.batch)
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        pred = F.softmax(x)

        loss = F.cross_entropy(pred, batch.y)
        self.log("training_loss", loss)
        self.log("training_accuracy", self.training_accuracy(pred, batch.y))
        return loss

    def validation_step(self, batch, idx):
        x = self.conv1(batch.x, batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index)
        x = gnn.global_mean_pool(x, batch.batch)
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        pred = F.softmax(x)

        loss = F.cross_entropy(pred, batch.y)
        self.log("validation_loss", loss)
        self.log("validation_accuracy", self.validation_accuracy(pred, batch.y))
