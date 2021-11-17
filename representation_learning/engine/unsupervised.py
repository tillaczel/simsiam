from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import torch

from representation_learning.models import get_encoder, get_predictor
from representation_learning.optimizer import get_optimizers, get_schedulers
from representation_learning.loss import symmetric_cos_dist
from representation_learning.metrics import Metrics


class UnsupervisedEngine(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        proj_conf, pred_conf = config.model.projector, config.model.predictor
        self.resnet, self.projector = get_encoder(n_p_layers=proj_conf.n_layers, emb_dim=proj_conf.emb_dim,
                                                  out_bn=proj_conf.out_bn)
        self.predictor = get_predictor(n_layers=pred_conf.n_layers, emb_dim=proj_conf.emb_dim,
                                       hid_dim=pred_conf.hid_dim, out_bn=pred_conf.out_bn)
        self.loss_func = symmetric_cos_dist
        self.automatic_optimization = False
        self.metrics = Metrics(config.dataset.n_classes, (1, 3, 5),
                               knn_k=config.evaluation.knn.knn_k, knn_t=config.evaluation.knn.knn_t)
        self.epoch_losses = list()
        self.full_eval_every_n = config.evaluation.knn.full_eval_every_n
        self.feature_bank_f, self.feature_bank_z, self.feature_bank_y = list(), list(), list()

    @property
    def lr(self):
        result = {
            'encoder_lr': self.optimizers()[0].param_groups[0]['lr'],
            'predictor_lr': self.optimizers()[1].param_groups[0]['lr']
        }
        return result

    @property
    def full_eval(self):
        eval_epoch = self.current_epoch % self.full_eval_every_n == 0
        last_epoch = self.config.training.max_epochs-1 == self.current_epoch
        return eval_epoch or last_epoch

    def forward(self, x):
        x = self.resnet(x)
        return x

    def training_step(self, batch, batch_idx):
        opt_enc, opt_pred = self.optimizers()
        opt_enc.zero_grad(), opt_pred.zero_grad()
        x, y, x1, x2 = batch
        f1, f2 = self.resnet(x1), self.resnet(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = self.loss_func(z1.detach(), z2.detach(), p1, p2)

        loss.backward()
        opt_enc.step(), opt_pred.step()

        self.log_dict(self.lr, prog_bar=True, on_step=True, logger=False)  # For progress bar
        self.epoch_losses.append(loss.detach().cpu().numpy())
        if self.full_eval:
            self.resnet.eval(), self.projector.eval()
            with torch.no_grad():
                f = self.resnet(x)
                z = self.projector(f).detach().cpu()
            self.feature_bank_f.append(f.detach().cpu()), self.feature_bank_z.append(z)
            self.feature_bank_y.append(y.detach().cpu())
            self.resnet.train(), self.projector.train()

    def training_epoch_end(self, outputs: list):
        loss = np.mean(self.epoch_losses)
        self.epoch_losses = list()
        metrics = {'train/loss': loss}
        metrics.update({f'train/{k}': v for k, v in self.lr.items()})
        self.logger.experiment.log(metrics, step=self.current_epoch)  # For wandb
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False, logger=False, sync_dist=True)  # For callbacks

        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self.resnet(x)
        z = self.projector(f)
        return f.detach().cpu(), z.detach().cpu(), y.detach().cpu()

    def validation_epoch_end(self, outputs: list):
        self.calc_acc(outputs, 'valid')

    def calc_acc(self, outputs, data_split):
        f, z, y = map(torch.cat, zip(*outputs))
        f, z, y = f.numpy(), z.numpy(), y.numpy()

        if self.full_eval:
            f_train, z_train = torch.cat(self.feature_bank_f).numpy(), torch.cat(self.feature_bank_z).numpy()
            y_train = torch.cat(self.feature_bank_y).numpy()
            self.feature_bank_f, self.feature_bank_z, self.feature_bank_y = list(), list(), list()
            _metrics = self.metrics.run(f, z, y, f_train, z_train, y_train)
        else:
            _metrics = self.metrics.run(f, z, y)
        metrics = dict()
        for k, v in _metrics.items():
            metrics[f'{data_split}/{k}'] = v

        self.logger.experiment.log(metrics, step=self.current_epoch)  # For wandb
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False, logger=False, sync_dist=True)  # For callbacks

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        f = self.resnet(x)
        z = self.projector(f)
        return f.detach().cpu(), z.detach().cpu(), y.detach().cpu()

    def configure_optimizers(self):
        optimizers = get_optimizers(self.config.training.optimizer, self.resnet, self.projector, self.predictor)
        training_config = self.config.training
        if training_config.scheduler is not None and \
                not (training_config.scheduler.encoder is None and training_config.scheduler.predictor is None):
            schedulers = get_schedulers(training_config, optimizers)
            return optimizers, schedulers
        else:
            return optimizers


