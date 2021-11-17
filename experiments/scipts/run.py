import sys
import os
sys.path.append(os.getcwd())

import hydra
from omegaconf import DictConfig, OmegaConf
from representation_learning.api import train, evaluate


@hydra.main(config_path='.', config_name="config")
def main(config: DictConfig):
    config = OmegaConf.structured(config)
    results = train(config)
    evaluate(results, config)


if __name__ == '__main__':
    main()
