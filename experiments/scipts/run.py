import sys
import os
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from representation_learning.api import train, evaluate


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Provided config not found - {config_path}.')

    config = OmegaConf.load(config_path)
    results = train(config)
    evaluate(results, config)
