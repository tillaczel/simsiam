from simsiam.data import get_unsupervised_dataloaders
from simsiam.engine import UnsupervisedEngine
from simsiam.trainer import create_trainer
from simsiam.utils import mkdir_if_missing


def make_dirs(config: DictConfig):
    with open_dict(config):
        config.experiment.exp_dir = os.path.join(os.getcwd(), 'experiments')
    mkdir_if_missing(config.experiment.exp_dir)
    mkdir_if_missing(config.experiment.results_dir)
    mkdir_if_missing(config.experiment.data_dir)
    return config


def train(config: DictConfig):
    config = make_dirs(config)

    pl.seed_everything(config.experiment.seed)
    train_dataloader, val_dataloader, test_dataloader, train_predict_dataloader\
        = get_unsupervised_dataloaders(config.dataset, config.experiment.data_dir)
    engine = UnsupervisedEngine(config)
    trainer = create_trainer(config)

    trainer.fit(engine, train_dataloader, val_dataloader)
    path = os.path.join(engine.logger.save_dir, engine.logger.experiment.project, engine.logger.experiment.id, 'model.ckpt')
    engine.trainer.save_checkpoint(path)
    wandb.save(path)

    outputs = trainer.predict(engine, train_predict_dataloader)
    f_train, z_train, y_train = map(np.concatenate, zip(*outputs))
    outputs = trainer.predict(engine, val_dataloader)
    f_valid, z_valid, y_valid = map(np.concatenate, zip(*outputs))
    outputs = trainer.predict(engine, test_dataloader)
    f_test, z_test, y_test = map(np.concatenate, zip(*outputs))
    return f_train, z_train, y_train, f_valid, z_valid, y_valid, f_test, z_test, y_test