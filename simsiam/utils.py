import os

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
