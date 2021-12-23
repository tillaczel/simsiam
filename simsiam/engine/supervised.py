from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import torch
import wandb

from simsiam.models import get_resnet
from simsiam.metrics import get_accuracy
from simsiam.optimizer import get_optimizer, get_scheduler


class SupervisedEngine(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.resnet = get_resnet(num_classes=config.dataset.n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.predict_step = self.validation_step
        self.test_step = self.validation_step

    @property
    def lr(self):
        result = self.optimizers().param_groups[0]['lr']
        return result

    def forward(self, x):
        x = self.resnet(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.resnet(x)
        loss = self.loss_func(y_hat, y[:, 0])

        self.log('lr', self.lr, prog_bar=True, on_step=True, logger=False)  # For progress bar
        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = {'train/loss': loss}
        metrics.update({f'train/lr': self.lr})
        self.logger.experiment.log(metrics, step=self.current_epoch)  # For wandb
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False, logger=False, sync_dist=True)  # For callbacks

    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.resnet(x)
        return f.detach().cpu(), y.detach().cpu()

    def validation_epoch_end(self, outputs: list):
        self.calc_acc(outputs, 'valid')

    def calc_acc(self, outputs, data_split):
        y_hat, y = map(torch.cat, zip(*outputs))
        y_hat, y = np.argsort(y_hat.numpy(), axis=1)[:, ::-1], y.numpy()
        acc = dict()
        _acc = get_accuracy(y_hat, y, (1, 3, 5))
        for k, v in _acc.items():
            acc[f'{data_split}/supervised_{k}'] = v
        self.logger.experiment.log(acc, step=self.current_epoch)  # For wandb
        self.log_dict(acc, prog_bar=False, on_epoch=True, on_step=False, logger=False,
                      sync_dist=True)  # For callbacks

    def configure_optimizers(self):
        training_config = self.config.training
        optimizer = get_optimizer(training_config.optimizer, self.resnet.parameters())
        if training_config.scheduler is not None:
            scheduler = get_scheduler(training_config.scheduler, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer





