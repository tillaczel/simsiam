import torch
from omegaconf import DictConfig


def get_optimizers(optim_config: DictConfig, encoder, predictor):
    enc_opt = get_optimizer(optim_config.encoder, encoder.parameters())
    pred_opt = get_optimizer(optim_config.predictor, predictor.parameters())
    return [enc_opt, pred_opt]


def get_optimizer(optim_config: DictConfig, params):
    name = optim_config.name
    lr = optim_config.lr

    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=optim_config.momentum, weight_decay=optim_config.weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f'{name} not in optimizers')


def get_schedulers(training_config, optimizers):
    result = list()
    if training_config.scheduler.encoder is not None:
        result.append(get_scheduler(training_config, training_config.scheduler.encoder, optimizers[0]))
    if training_config.scheduler.predictor is not None:
        result.append(get_scheduler(training_config, training_config.scheduler.predictor, optimizers[1]))
    return result


def get_scheduler(training_config, scheduler_config, optimizer):
    name = scheduler_config.name
    if name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=training_config.max_epochs)
        return dict(scheduler=scheduler)
    else:
        raise ValueError(f'{name} not in schedulers')

