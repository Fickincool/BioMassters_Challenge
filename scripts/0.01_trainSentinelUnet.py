from bioMass.dataloader import SentinelDataModule
from bioMass.model import Unet
from pytorch_lightning import Trainer
from torch import nn
import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import numpy as np
import os

from argparse import ArgumentParser


class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            if trainer.model.image in ['image_s1', 'image_s2']:
                image, target = batch[trainer.model.image], batch['label']
            elif trainer.model.image == 'image_s1+s2':
                image = torch.concat([batch['image_s1'], batch['image_s2']], axis=1)
                target = batch['label']

            logger = trainer.logger
            print(image.shape)
            logger.experiment.add_histogram("input", image, global_step=trainer.global_step)
            # logger.experiment.add_histogram("target", target, global_step=trainer.global_step)


def parse_logdir(in_channels, tensorboard_logdir='/home/ubuntu/Thesis/backup_data/bioMass_data/model_logs/overfitSample/'):
    if in_channels==15:
        tensorboard_logdir = os.path.join(tensorboard_logdir, 'UNet_full')
    elif in_channels==11:
        tensorboard_logdir = os.path.join(tensorboard_logdir, 'UNet_S2')
    elif in_channels==4:
        tensorboard_logdir = os.path.join(tensorboard_logdir, 'UNet_S1')
    
    return tensorboard_logdir

def main(args):

    for in_channels in [4, 11, 15]:    
        logger = pl_loggers.TensorBoardLogger(
            parse_logdir(in_channels=in_channels), name="", default_hp_metric=False
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

        model = Unet(loss_fn=loss_module, lr=1e-4, init_n_features=64, in_channels=in_channels, dropout_p=dropout_p)

        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)