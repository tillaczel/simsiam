import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb

from representation_learning.data.__init__ import get_dataloaders, LinearDataset
from representation_learning.engine import UnsupervisedEngine, LinearEngine
from representation_learning.trainer import create_trainer
from representation_learning.utils import mkdir_if_missing


def update_train_config(config: DictConfig):
    exp_dir = mkdir_if_missing(os.path.join(os.getcwd(), 'experiments'))
    results_dir = mkdir_if_missing(os.path.join(exp_dir, 'results'))
    config.experiment['save_dir'] = mkdir_if_missing(os.path.join(results_dir, config.experiment.name))
    config.experiment['data_dir'] = mkdir_if_missing(os.path.join(exp_dir, 'data'))
    return config


def train(config: DictConfig):
    config = update_train_config(config)

    pl.seed_everything(config.experiment.seed)
    train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader\
        = get_dataloaders(config.dataset, config.experiment.data_dir)
    engine = UnsupervisedEngine(config)
    trainer = create_trainer(config)

    trainer.fit(engine, train_dataloader, val_dataloader)

    outputs = trainer.predict(engine, train_predict_dataloader)
    f_train, y_train = map(torch.cat, zip(*outputs))
    outputs = trainer.predict(engine, val_dataloader)
    f_valid, y_valid = map(torch.cat, zip(*outputs))
    outputs = trainer.predict(engine, test_dataloader)
    f_test, y_test = map(torch.cat, zip(*outputs))
    return f_train, y_train, f_valid, y_valid, f_test, y_test


def evaluate(results, config: DictConfig):
    train_dataloader, valid_dataloader, test_dataloader = get_linear_dataloaders(config.evaluation.linear, *results)

    engine = LinearEngine(config.evaluation.linear.in_dim, config.dataset.n_classes, config.evaluation.linear.lr,
                          config.evaluation.linear.momentum, config.evaluation.linear.weight_decay,
                          config.evaluation.linear.max_epochs)
    trainer = pl.Trainer(max_epochs=config.evaluation.linear.max_epochs,
                         deterministic=True,
                         terminate_on_nan=True,
                         num_sanity_val_steps=0
                         )
    trainer.fit(engine, train_dataloader, valid_dataloader)
    trainer.test(engine, test_dataloader)


def get_linear_dataloaders(config, f_train, y_train, f_valid, y_valid, f_test, y_test):
    # np.savetxt('z_train.csv', z_train, delimiter=',')
    # np.savetxt('y_train.csv', y_train, delimiter=',')
    # np.savetxt('z_valid.csv', z_valid, delimiter=',')
    # np.savetxt('y_valid.csv', y_valid, delimiter=',')
    # np.savetxt('z_test.csv', z_test, delimiter=',')
    # np.savetxt('y_test.csv', y_test, delimiter=',')

    train_set = LinearDataset(f_train, y_train)
    valid_set = LinearDataset(f_valid, y_valid)
    test_set = LinearDataset(f_test, y_test)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=True, shuffle=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=False, shuffle=False, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                 drop_last=False, shuffle=False, persistent_workers=True)
    return train_dataloader, valid_dataloader, test_dataloader


