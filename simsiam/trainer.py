from omegaconf import DictConfig, OmegaConf
import wandb
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, progress


def crecte_ckpt_callback(ckpt_config: DictConfig):
    return ModelCheckpoint(
        save_top_k=ckpt_config.save_top_k,
        monitor=ckpt_config.monitor,
        mode=ckpt_config.mode,
    )


def create_logger(config: DictConfig):
    wandb.init(project='representation_learning')
    wandb.run.name = config.experiment.name
    wandb.save('*.ckpt')
    logger = loggers.WandbLogger(project='representation_learning', save_dir=config.experiment.results_dir,
                                 config=config, log_model='all')
    return logger


def create_trainer(config: DictConfig):
    logger = create_logger(config)
    _callbacks = [crecte_ckpt_callback(config.training.ckpt_callback)]
    trainer = pl.Trainer(logger=logger,
                         gpus=config.experiment.gpu,
                         max_epochs=config.training.max_epochs,
                         deterministic=True,
                         detect_anomaly=True,
                         num_sanity_val_steps=0,
                         callbacks=_callbacks
                         )
    return trainer


