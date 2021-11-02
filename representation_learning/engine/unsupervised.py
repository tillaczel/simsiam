from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import torch

from representation_learning.models import get_encoder, get_predictor
from representation_learning.optimizer import get_optimizer, get_scheduler
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

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        x = self.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = self.loss_func(z1.detach(), z2.detach(), p1, p2)
        self.log('lr', self.lr, prog_bar=True, on_step=True, logger=False)  # For callbacks
        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        loss = np.mean(list(map(lambda x: x['loss'].detach().cpu().numpy(), outputs)))
        metrics = {
            'train/lr': self.lr,
            'train/loss': loss
        }
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
        optimizer = get_optimizer(self.config.training.optimizer, self.parameters())
        scheduler_config = self.config.training.scheduler
        if scheduler_config is not None:
            scheduler = get_scheduler(scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer


