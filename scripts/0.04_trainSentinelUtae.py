from bioMass.dataloader2 import SentinelDataModule
from bioMass.model import Utae
from pytorch_lightning import Trainer
from torch import nn
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import numpy as np
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from argparse import ArgumentParser

def parse_logdir(
    tensorboard_logdir='/home/ubuntu/Thesis/backup_data/bioMass_data/model_logs/overfitSample/'
    ):

    tensorboard_logdir = os.path.join(tensorboard_logdir, 'Utae')
    return tensorboard_logdir

def update_hparams(model, param_name, param):
    fname = os.path.join(model.logger.log_dir, 'hparams.yaml')

    with open(fname) as f:
        newdct = yaml.load(f, Loader=yaml.BaseLoader)

    newdct[param_name] = param

    with open(fname, "w") as f:
        yaml.dump(newdct, f)

    return

dm_params = {
    'is_train':True, 'max_chips':555, 'loader_device':'cpu', 'num_workers':24,
    'split_proportions':[0.8, 0.2, 0], 'batch_size':10
    }

def main(args):


    logger = pl_loggers.TensorBoardLogger(
        parse_logdir(), name="", default_hp_metric=False
    )

    # Parse initial arguments
    trainer = Trainer.from_argparse_args(args)
    trainer.logger = logger

    early_stop_callback = EarlyStopping(
            monitor="Val RMSE",
            min_delta=1e-4,
            patience=50,
            verbose=True,
            mode="min",
        )
    
    ckpt = ModelCheckpoint(dirpath=None, save_top_k=1, monitor="Val RMSE")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, lr_monitor, ckpt]
    trainer.callbacks = callbacks

    loss_module = nn.MSELoss(reduction='mean')

    dm = SentinelDataModule(**dm_params)

    model = Utae(loss_fn=loss_module, lr=1e-4, input_dim=15)

    trainer.fit(model, datamodule=dm)

    update_hparams(model, 'dm_params', dm_params)
    update_hparams(model, 'trainer_params', vars(args))
    
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)