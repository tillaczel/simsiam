from omegaconf import DictConfig, OmegaConf
import wandb
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, progress


def crecte_ckpt_callback(ckpt_config: DictConfig):
    return ModelCheckpoint(
        filename='epoch',
        save_top_k=ckpt_config.save_top_k,
        monitor=ckpt_config.monitor,
        mode=ckpt_config.mode,
    )


def create_logger(config: DictConfig):
    logger = loggers.WandbLogger(project='representation_learning', save_dir=config.experiment.save_dir, config=config,
                                 log_model='all')
    #import pdb; pdb.set_trace()
    #cfg_path = os.path.join(logger.save_dir, 'train_config.yaml')
    #with open(cfg_path, 'w') as fh:
    #    fh.write(OmegaConf.to_yaml(config))
    #wandb.save(cfg_path, base_path=logger.save_dir)  # this will force sync it
    #wandb.save('*.ckpt')  # should keep it up to date
    return logger


def create_trainer(config: DictConfig):
    logger = create_logger(config)
    _callbacks = [progress.ProgressBar(), crecte_ckpt_callback(config.training.ckpt_callback)]
    trainer = pl.Trainer(logger=logger,
                         gpus=config.experiment.gpu,
                         max_epochs=config.training.max_epochs,
                         progress_bar_refresh_rate=20,
                         deterministic=True,
                         terminate_on_nan=True,
                         num_sanity_val_steps=0,
                         callbacks=_callbacks
                         )
    return trainer


