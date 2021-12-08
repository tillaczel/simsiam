from omegaconf import DictConfig
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import datasets
import numpy as np
import os

from simsiam.data.tranforms import get_transforms


def get_datasets(dataset, data_dir, val_split, train_split_ratio, name, normalize_bool=True):
    train_transform, test_transform = get_transforms(name, normalize_bool)

    _test_set = dataset(data_dir, train=False, download=True)

    _train_set = dataset(data_dir, train=True, download=True)
    if val_split:
        train_len = int(len(_train_set) * train_split_ratio)
        valid_len = len(_train_set)-train_len
        _train_set, _valid_set = random_split(_train_set, [train_len, valid_len])
    else:
        _valid_set = _test_set

    train_set = DoubleAugmentDataset(_train_set, train_transform, test_transform)
    valid_set = AugmentDataset(_valid_set, test_transform)
    test_set = AugmentDataset(_test_set, test_transform)
    train_set_predict = AugmentDataset(_train_set, test_transform)
    return train_set, valid_set, test_set, train_set_predict


def get_unsupervised_dataloaders(data_cfg: DictConfig, data_dir: str, normalize_bool=True):
    batch_size = data_cfg.batch_size
    train_split_ratio = data_cfg.train_split_ratio
    num_workers = data_cfg.num_workers
    val_split = data_cfg.val_split

    train_set, valid_set, test_set, train_set_predict = \
        get_datasets(datasets.CIFAR10, data_dir, val_split, train_split_ratio, data_cfg.name, normalize_bool)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True, shuffle=True, persistent_workers=True)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers,
                                drop_last=False, shuffle=False, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                                 drop_last=False, shuffle=False, persistent_workers=True)
    train_predict_dataloader = DataLoader(train_set_predict, batch_size=batch_size,
                                          num_workers=num_workers,  drop_last=False, shuffle=False,
                                          persistent_workers=False)
    return train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader


class DoubleAugmentDataset(Dataset):
    def __init__(self, dataset, transform, test_transform):
        self.dataset = dataset
        self.transform = transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x0 = self.test_transform(x)
        x1, x2 = self.transform(x), self.transform(x)
        return x0, torch.Tensor([y]), x1, x2


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


class LinearDataset(Dataset):
    def __init__(self, f, z, y):
        self.f, self.z, self.y = f, z, y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        f, z = torch.FloatTensor(self.f[index]), torch.FloatTensor(self.f[index])
        y = torch.LongTensor([int(self.y[index])])
        return f, z, y


def get_linear_dataloaders(config, f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test):
    train_set = LinearDataset(f_train, z_train, y_train)
    valid_set = LinearDataset(f_valid, z_valid, y_valid)
    test_set = LinearDataset(f_test, z_test, y_test)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=False, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                 drop_last=False, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader
