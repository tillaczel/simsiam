import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from representation_learning.metrics import get_accuracy
import wandb


class LinearEngine(pl.LightningModule):

    def __init__(self, embedding_dim, n_classes, lr, momentum, weight_decay, max_epoch):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.lr, self.momentum, self.weight_decay, self.max_epoch = lr, momentum, weight_decay, max_epoch

        self.model = nn.Linear(embedding_dim, n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        f, y = batch
        y_hat = self.model(f)
        loss = self.criterion(y_hat, y[:, 0])
        return loss

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        f, y = batch
        y_hat = self.model(f)
        y_hat = torch.argsort(y_hat, descending=True)
        return y_hat.detach().cpu(), y

    def validation_epoch_end(self, outputs: list):
        self.calc_acc(outputs, 'valid')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: list):
        self.calc_acc(outputs, 'test')

    def calc_acc(self, outputs, data_split):
        y_hat, y = map(torch.cat, zip(*outputs))
        acc = dict()
        _acc = get_accuracy(y_hat.numpy(), y.numpy(), (1, 3, 5))
        for k, v in _acc.items():
            acc[f'linear/{data_split}_{k}'] = v
        wandb.log(acc, step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        return [optimizer], [scheduler]
