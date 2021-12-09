import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np
import wandb

from simsiam.data import get_unsupervised_dataloaders, get_supervised_dataloaders
from simsiam.engine import UnsupervisedEngine, SupervisedEngine
from simsiam.trainer import create_trainer
from simsiam.utils import make_dirs


def train(config: DictConfig):
    config = make_dirs(config)

    pl.seed_everything(config.experiment.seed)
    if config.experiment.supervised:
        engine = SupervisedEngine(config)
        train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader \
            = get_supervised_dataloaders(config.dataset, config.experiment.data_dir)
    else:
        engine = UnsupervisedEngine(config)
        train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader \
            = get_unsupervised_dataloaders(config.dataset, config.experiment.data_dir)
    trainer = create_trainer(config)

    trainer.fit(engine, train_dataloader, val_dataloader)
    path = os.path.join(engine.logger.save_dir, engine.logger.experiment.project, engine.logger.experiment.id, 'model.ckpt')
    engine.trainer.save_checkpoint(path)
    wandb.save(path)

    outputs = trainer.predict(engine, train_predict_dataloader)
    f_train, z_train, y_train = map(np.concatenate, zip(*outputs))
    outputs = trainer.predict(engine, val_dataloader)
    f_valid, z_valid, y_valid = map(np.concatenate, zip(*outputs))
    outputs = trainer.predict(engine, test_dataloader)
    f_test, z_test, y_test = map(np.concatenate, zip(*outputs))
    return f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test

