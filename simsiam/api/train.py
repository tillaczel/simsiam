import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np
import wandb

from simsiam.data import get_unsupervised_dataloaders, get_supervised_dataloaders
from simsiam.engine import UnsupervisedEngine, SupervisedEngine
from simsiam.trainer import create_trainer
from simsiam.utils import make_dirs


def train_unsupervised(config: DictConfig):
    config = setup(config)
    engine = UnsupervisedEngine(config)
    train_dataloader, val_dataloader, train_predict_dataloader \
        = get_unsupervised_dataloaders(config, config.experiment.data_dir)
    engine, trainer = train(config, engine, train_dataloader, val_dataloader)

    outputs = trainer.predict(engine, train_predict_dataloader)
    f_train, z_train, y_train = map(np.concatenate, zip(*outputs))
    outputs = trainer.predict(engine, val_dataloader)
    f_valid, z_valid, y_valid = map(np.concatenate, zip(*outputs))
    return f_train, z_train, y_train, f_valid, z_valid, y_valid


def train_supervised(config: DictConfig):
    config = setup(config)
    engine = SupervisedEngine(config)
    train_dataloader, val_dataloader, train_predict_dataloader \
        = get_supervised_dataloaders(config, config.experiment.data_dir)
    return train(config, engine, train_dataloader, val_dataloader)


def train(config: DictConfig, engine, train_dataloader, val_dataloader):
    trainer = create_trainer(config)

    trainer.fit(engine, train_dataloader, val_dataloader)
    path = os.path.join(engine.logger.save_dir, engine.logger.experiment.project, engine.logger.experiment.id, 'model.ckpt')
    engine.trainer.save_checkpoint(path)
    wandb.save(path)
    return engine, trainer


def setup(config: DictConfig):
    pl.seed_everything(config.experiment.seed)
    config = make_dirs(config)
    return config
