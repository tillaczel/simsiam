import sys
import os
sys.path.append(os.getcwd())

import hydra
from omegaconf import DictConfig, OmegaConf
from simsiam.api import train_supervised, evaluate


@hydra.main(config_path='.', config_name="config")
def main(config: DictConfig):
    config = OmegaConf.structured(config)
    results = train_supervised(config)
    print(evaluate(results, config))


if __name__ == '__main__':
    main()
