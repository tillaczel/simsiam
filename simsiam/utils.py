import os
import numpy as np

from omegaconf import DictConfig, open_dict
from hydra.utils import get_original_cwd


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def make_dirs(config: DictConfig):
    with open_dict(config):
        config.experiment.exp_dir = mkdir_if_missing(os.path.join(get_original_cwd(), 'experiments'))
        config.experiment.results_dir = mkdir_if_missing(os.path.join(config.experiment.exp_dir, 'results'))
        config.experiment.data_dir = mkdir_if_missing(os.path.join(config.experiment.exp_dir, 'data'))
    return config


def get_subset_idx(subset, exp_dir, val_split):
    if val_split:
        idx = np.genfromtxt(os.path.join(exp_dir, 'shuffle_index', 'idx_40000.csv'),
                            delimiter=',')
    else:
        idx = np.genfromtxt(os.path.join(exp_dir, 'shuffle_index', 'idx_50000.csv'),
                            delimiter=',')
    return idx[:int(idx.shape[0] * subset / 100)].astype(np.int32)
