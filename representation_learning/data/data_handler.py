import os
from omegaconf import DictConfig
import pytorch_lightning as pl
from typing import Optional
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import datasets

from representation_learning.data.tranforms import get_transforms


class DataHandler(pl.LightningDataModule):
    def __init__(self, dataset, data_cfg: DictConfig, data_dir: str):
        super().__init__()
        self.dataset = dataset
        self.batch_size = data_cfg.batch_size
        self.train_split_ratio = data_cfg.train_split_ratio
        self.crop_size = data_cfg.crop_size
        self.num_workers = data_cfg.num_workers

        self.data_dir = data_dir

        self.train_transform, self.test_transform = get_transforms(self.crop_size)
        self.train_set, self.valid_set, self.test_set = self._init_datasets()

    def _init_datasets(self):
        test_set = self.dataset(self.data_dir, train=False, download=True)

        datast_full = self.dataset(self.data_dir, train=True, download=True)
        train_len = int(len(datast_full) * self.train_split_ratio)
        valid_len = len(datast_full)-train_len
        train_set, valid_set = random_split(datast_full, [train_len, valid_len])

        train_set = DoubleAugmentDataset(train_set, self.train_transform)
        valid_set = AugmentDataset(valid_set, self.test_transform)
        test_set = AugmentDataset(test_set, self.test_transform)
        return train_set, valid_set, test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)


class DoubleAugmentDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        x1, x2 = self.transform(x), self.transform(x)
        return x1, x2


class AugmentDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.transform(x)
        return x, torch.Tensor([y])


DATASETS = {
    'CIFAR10': datasets.CIFAR10
}


def data_handler_factory(data_cfg: DictConfig, data_dir: str):
    name = data_cfg.name
    if name in set(DATASETS.keys()):
        return DataHandler(DATASETS[name], data_cfg, data_dir)
    raise ValueError(f'Dataset {name} not in available datasets: {DATASETS.keys()}')
