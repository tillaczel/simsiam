import torch
from omegaconf import DictConfig


def get_optimizer(optim_config: DictConfig, params):
    name = optim_config.name
    lr = optim_config.lr

    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f'{name} not in optimizers')


def get_scheduler(scheduler_config, optimizer):
    name = scheduler_config.name
    monitor = scheduler_config.monitor

    if name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=scheduler_config.mode,
                                                               patience=scheduler_config.patience,
                                                               factor=scheduler_config.factor,
                                                               min_lr=scheduler_config.min_lr)
        return dict(scheduler=scheduler, monitor=monitor)
    else:
        raise ValueError(f'{name} not in schedulers')
