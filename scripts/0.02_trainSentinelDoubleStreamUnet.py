from bioMass.dataloader import SentinelDataModule
from bioMass.model import DoubleStreamUnet
from pytorch_lightning import Trainer
from torch import nn
import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import numpy as np
import os

from argparse import ArgumentParser



def parse_logdir(tensorboard_logdir='/home/ubuntu/Thesis/backup_data/bioMass_data/model_logs/overfitSample/'):
    tensorboard_logdir = os.path.join(tensorboard_logdir, 'DoubleStreamUNet')
    return tensorboard_logdir

def main(args):
  
    logger = pl_loggers.TensorBoardLogger(
        parse_logdir(), name="", default_hp_metric=False
    )

    # Parse initial arguments
    trainer = Trainer.from_argparse_args(args)
    trainer.logger = logger
    # trainer.callbacks = [InputMonitor()]

    loss_module = nn.MSELoss(reduction='mean')

    dm = SentinelDataModule(max_chips=50, loader_device='cpu', num_workers=1)

    # dont use dropout if we are trying to overfit
    if vars(args)['overfit_batches']==1:
        dropout_p = 0
    else:
        dropout_p = 0.5

    model = DoubleStreamUnet(loss_fn=loss_module, lr=1e-4, dropout_p=dropout_p)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)