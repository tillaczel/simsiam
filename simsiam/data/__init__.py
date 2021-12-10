from omegaconf import DictConfig
from torch.utils.data import random_split, DataLoader
import torchvision

from simsiam.data.datasets import DoubleAugmentDataset, AugmentDataset, LinearDataset
from simsiam.data.tranforms import get_unsupervised_transforms, get_supervised_transforms
from simsiam.utils import get_subset_idx


def get_datasets(dataset, data_dir, val_split, train_split_ratio):
    _test_set = dataset(data_dir, train=False, download=True)

    _train_set = dataset(data_dir, train=True, download=True)
    if val_split:
        train_len = int(len(_train_set) * train_split_ratio)
        valid_len = len(_train_set)-train_len
        _train_set, _valid_set = random_split(_train_set, [train_len, valid_len])
    else:
        _valid_set = _test_set

    return _train_set, _valid_set


def get_dataloaders(train_set, valid_set, test_set, train_set_predict, batch_size, num_workers):
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


def get_unsupervised_dataloaders(config: DictConfig, data_dir: str, normalize_bool=True):
    data_cfg = config.dataset
    batch_size = data_cfg.batch_size
    train_split_ratio = data_cfg.train_split_ratio
    num_workers = data_cfg.num_workers
    val_split = data_cfg.val_split

    _train_set, _valid_set = \
        get_datasets(torchvision.datasets.CIFAR10, data_dir, val_split, train_split_ratio)
    train_transform, test_transform = get_unsupervised_transforms(data_cfg.name, normalize_bool)
    train_set = DoubleAugmentDataset(_train_set, train_transform, test_transform)
    valid_set = AugmentDataset(_valid_set, test_transform)
    train_set_predict = AugmentDataset(_train_set, test_transform)
    return get_dataloaders(train_set, valid_set, train_set_predict, batch_size, num_workers)


def get_supervised_dataloaders(config: DictConfig, data_dir: str, normalize_bool=True):
    data_cfg = config.dataset
    batch_size = data_cfg.batch_size
    train_split_ratio = data_cfg.train_split_ratio
    num_workers = data_cfg.num_workers
    val_split = data_cfg.val_split

    _train_set, _valid_set = \
        get_datasets(torchvision.datasets.CIFAR10, data_dir, val_split, train_split_ratio)
    idx = get_subset_idx(data_cfg.subset, config.experiment.exp_dir, config.dataset.val_split)
    _train_set_subset = list()
    for i in idx:
        _train_set_subset.append(_train_set[i])
    _train_set_subset = _train_set_subset*int(100/data_cfg.subset)

    train_transform, test_transform = get_supervised_transforms(data_cfg.name, normalize_bool)
    train_set = AugmentDataset(_train_set_subset, train_transform)
    valid_set = AugmentDataset(_valid_set, test_transform)
    train_set_predict = AugmentDataset(_train_set_subset, test_transform)

    return get_dataloaders(train_set, valid_set, train_set_predict, batch_size, num_workers)


def get_linear_dataloaders(config, f_train, z_train, y_train, f_valid, z_valid, y_valid):
    train_set = LinearDataset(f_train, z_train, y_train)
    valid_set = LinearDataset(f_valid, z_valid, y_valid)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=False, shuffle=False)
    return train_dataloader, valid_dataloader
