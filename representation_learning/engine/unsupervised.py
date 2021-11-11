import pdb

from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import torch

from representation_learning.models import get_encoder, get_predictor
from representation_learning.optimizer import get_optimizers, get_schedulers
from representation_learning.loss import symmetric_cos_dist
from representation_learning.metrics import knn_acc


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        proj_conf, pred_conf = config.model.projector, config.model.predictor
        self.encoder = get_encoder(n_p_layers=proj_conf.n_layers, emb_dim=proj_conf.emb_dim, out_bn=proj_conf.out_bn)
        self.predictor = get_predictor(n_layers=pred_conf.n_layers, emb_dim=proj_conf.emb_dim,
                                       hid_dim=pred_conf.hid_dim, out_bn=pred_conf.out_bn)
        self.loss_func = symmetric_cos_dist
        self.automatic_optimization = False
        self.epoch_losses = list()

    @property
    def lr(self):
        result = {
            'encoder_lr': self.optimizers()[0].param_groups[0]['lr'],
            'predictor_lr': self.optimizers()[1].param_groups[0]['lr']
        }
        return result

    def forward(self, x):
        x = self.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        opt_enc, opt_pred = self.optimizers()
        opt_enc.zero_grad(), opt_pred.zero_grad()
        x1, x2 = batch
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = self.loss_func(z1.detach(), z2.detach(), p1, p2)

        loss.backward()
        opt_enc.step(), opt_pred.step()

        self.log_dict(self.lr, prog_bar=True, on_step=True, logger=False)  # For progress bar
        self.epoch_losses.append(loss.detach().cpu().numpy())

    def training_epoch_end(self, outputs: list):
        scheduler = self.lr_schedulers()
        scheduler.step()

        loss = np.mean(self.epoch_losses)
        self.epoch_losses = list()
        metrics = {'train/loss': loss}
        metrics.update({f'train/{k}': v for k, v in self.lr.items()})
        self.logger.experiment.log(metrics, step=self.current_epoch)  # For wandb
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False, logger=False)  # For callbacks

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        return z.detach().cpu(), y.detach().cpu()

    def validation_epoch_end(self, outputs: list):
        z, y = map(torch.cat, zip(*outputs))
        z, y = z.numpy(), y.numpy()
        acc = knn_acc(z, y, (1, 3, 5))
        metrics = dict()
        for k, v in acc.items():
            metrics[f'valid/{k}'] = v
        self.logger.experiment.log(metrics, step=self.current_epoch)  # For wandb
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False, logger=False)  # For callbacks

    def configure_optimizers(self):
        optimizers = get_optimizers(self.config.training.optimizer, self.encoder, self.predictor)
        training_config = self.config.training
        if training_config.scheduler is not None and \
                not (training_config.scheduler.encoder is None and training_config.scheduler.predictor is None):
            schedulers = get_schedulers(training_config, optimizers)
            return optimizers, schedulers
        else:
            return optimizers


