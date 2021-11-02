import os
import pytorch_lightning as pl
from omegaconf import  DictConfig

from representation_learning.data.data_handler import data_handler_factory
from representation_learning.engine.unsupervised import EngineModule
from representation_learning.trainer import create_trainer
from representation_learning.utils import mkdir_if_missing


def update_train_config(config: DictConfig):
    results_dir = mkdir_if_missing(os.path.join(os.getcwd(), 'experiments/results'))
    config.experiment['save_dir'] = mkdir_if_missing(os.path.join(results_dir, config.experiment.name))
    return config


def train(config: DictConfig):
    config = update_train_config(config)

    pl.seed_everything(config.experiment.seed)
    data_handler = data_handler_factory(config.dataset)
    engine = EngineModule(config)
    trainer = create_trainer(config)

    trainer.fit(model=engine, datamodule=data_handler)
