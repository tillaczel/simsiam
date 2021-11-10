import torch
from omegaconf import DictConfig


def get_optimizer(optim_config: DictConfig, params):
    name = optim_config.name
    lr = optim_config.lr

    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=optim_config.momentum, weight_decay=optim_config.weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f'{name} not in optimizers')


def get_scheduler(training_config, optimizer):
    scheduler_config = training_config.scheduler
    name = scheduler_config.name

    if name == 'plateau':
        monitor = scheduler_config.monitor
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=scheduler_config.mode,
                                                               patience=scheduler_config.patience,
                                                               factor=scheduler_config.factor,
                                                               min_lr=scheduler_config.min_lr)
        return dict(scheduler=scheduler, monitor=monitor)
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=training_config.max_epochs)
        return dict(scheduler=scheduler)
    else:
        raise ValueError(f'{name} not in schedulers')

