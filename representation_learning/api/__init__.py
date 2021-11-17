import os
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch

from representation_learning.data import get_unsupervised_dataloaders, get_linear_dataloaders
from representation_learning.engine import UnsupervisedEngine, LinearEngine
from representation_learning.trainer import create_trainer
from representation_learning.utils import mkdir_if_missing
from representation_learning.metrics import Metrics


def make_dirs(config: DictConfig):
    mkdir_if_missing(config.experiment.exp_dir)
    mkdir_if_missing(config.experiment.results_dir)
    mkdir_if_missing(config.experiment.data_dir)
    return config


def train(config: DictConfig):
    config = make_dirs(config)

    pl.seed_everything(config.experiment.seed)
    train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader\
        = get_unsupervised_dataloaders(config.dataset, config.experiment.data_dir)
    engine = UnsupervisedEngine(config)
    trainer = create_trainer(config)

    trainer.fit(engine, train_dataloader, val_dataloader)

    outputs = trainer.predict(engine, train_predict_dataloader)
    f_train, z_train, y_train = map(torch.cat, zip(*outputs))
    outputs = trainer.predict(engine, val_dataloader)
    f_valid, z_valid, y_valid = map(torch.cat, zip(*outputs))
    outputs = trainer.predict(engine, test_dataloader)
    f_test, z_test, y_test = map(torch.cat, zip(*outputs))
    return f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test


def evaluate(results, config: DictConfig):
    f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test = results
    train_dataloader, valid_dataloader, test_dataloader = \
        get_linear_dataloaders(config.evaluation.linear,
                               f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test)

    metrics = Metrics(config.dataset.n_classes, (1, 3, 5), knn_k=config.evaluation.knn.knn_k,
                      knn_t=config.evaluation.knn.knn_t)
    _metrics = metrics.run(f_test, z_test, y_test, f_train, z_train, y_train)
    metrics = dict()
    for k, v in _metrics.items():
        metrics[f'test/{k}'] = v

    engine = LinearEngine(config.evaluation.linear.in_dim, config.dataset.n_classes, config.evaluation.linear.lr,
                          config.evaluation.linear.momentum, config.evaluation.linear.weight_decay,
                          config.evaluation.linear.max_epochs)
    trainer = pl.Trainer(max_epochs=config.evaluation.linear.max_epochs,
                         deterministic=True,
                         terminate_on_nan=True,
                         num_sanity_val_steps=0,
                         gpus=config.experiment.gpu
                         )
    trainer.fit(engine, train_dataloader, valid_dataloader)
    trainer.test(engine, test_dataloader)


