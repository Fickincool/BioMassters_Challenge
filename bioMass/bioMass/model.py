import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# This is just a copy from the original implementation in:
# https://github.com/NVIDIA/partialconv/tree/master/models
from bioMass.partialConv2D import PartialConv2d

class _Unet(pl.LightningModule):
    "Pilot model"
    def __init__(self, init_n_features, in_channels, dropout_p):
        """Expected input: [B, C, S, S] where B the batch size, C input channels and S the image length.
        The data values are expected to be standardized and [0, 1] scaled.
        
        - p: dropout probability
        """

        super().__init__()
        self.init_n_features = init_n_features
        self.in_channels = in_channels
        self.p = dropout_p

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)

        self.EB0 = self.partialConv2d_combo(self.in_channels, self.init_n_features)
        self.EB1 = self.partialConv2d_combo(init_n_features, init_n_features*2)
        self.EB2 = self.partialConv2d_combo(init_n_features*2, init_n_features*4)
        self.EB3 = self.partialConv2d_combo(init_n_features*4, init_n_features*8)

        self.Bottom = self.partialConv2d_combo(init_n_features*8, init_n_features*8)

        self.DB2 = self.conv2d_combo(init_n_features*16, init_n_features*4)
        self.DB1 = self.conv2d_combo(init_n_features*8, init_n_features*2)
        self.DB0 = self.conv2d_combo(init_n_features*4, init_n_features)

        self.out = nn.Sequential(
            self.conv2d_combo(init_n_features*2, init_n_features), 
            nn.Conv2d(init_n_features, 1, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.1)
            )

        return

    def forward(self, x: torch.Tensor):
        "Input tensor of shape [batch_size, channels, side, side]"

        ##### ENCODER #####
        e0 = self.EB0(x) # 1, channels_out = 64
        # print('E0: ', e0.shape)
        e1 = self.EB1(self.maxpool(e0)) # 1/2, channels_out = 128
        # print('E1: ', e1.shape)
        e2 = self.EB2(self.maxpool(e1)) # 1/4, channels_out = 256
        # print('E2: ', e2.shape)
        e3 = self.EB3(self.maxpool(e2)) # 1/8, channels_out = 512
        # print('E3: ', e3.shape)
        
        ##### DECODER #####
        d3 = self.up(self.Bottom(self.maxpool(e3)))
        d3 = torch.concat([d3, e3], axis=1) # 1/8, channels_out = 1024
        # print('D3: ', d3.shape)
        d2 = self.up(self.DB2(d3))
        d2 = torch.concat([d2, e2], axis=1) # 1/4, channels_out = 512
        # print('D2: ', d2.shape)
        d1 = self.up(self.DB1(d2))
        d1 = torch.concat([d1, e1], axis=1) # 1/2, channels_out = 256
        # print('D1: ', d1.shape)
        d0 = self.up(self.DB0(d1))
        d0 = torch.concat([d0, e0], axis=1) # 1, channels_out = 128
        # print('D0: ', d0.shape)
 
        x = self.out(d0) # 1, channels_out = 1
        # print('Output shape: ', x.shape)

        return x

    def partialConv2d_combo(self, n_features_in, n_features_out):
        combo = nn.Sequential(
            PartialConv2d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            PartialConv2d(n_features_out, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return combo

    def conv2d_combo(self, n_features_in, n_features_out):
        combo = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv2d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return combo

class Unet(_Unet):
    def __init__(self, loss_fn, lr, init_n_features, in_channels, dropout_p):
        """Expected input: [B, C, S, S] where B the batch size, C input channels and S the image length.
        The data values are expected to be Normalized.
        
        - p: dropout probability
        """
        super().__init__(init_n_features, in_channels, dropout_p)
        self.lr = lr
        self.loss_fn = loss_fn

        # get type of image to use depending on the number of channels
        if in_channels==11:
            self.image = 'image_s2'
        elif in_channels==4:
            self.image = 'image_s1'
        elif in_channels==15:
            self.image = 'image_s1+s2'

        self.save_hyperparameters()

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        factor = 0.1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    "min",
                    verbose=True,
                    patience=10,
                    min_lr=1e-8,
                    factor=factor,
                ),
                "monitor": "Val RMSE",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch, batch_idx):
        if self.image in ['image_s1', 'image_s2']:
            image, target = batch[self.image], batch['label']
        elif self.image == 'image_s1+s2':
            image = torch.concat([batch['image_s1'], batch['image_s2']], axis=1)
            target = batch['label']

        pred = self(image)
        loss = self.loss_fn(pred, target)

        self.log(
            "Train RMSE",
            torch.round(torch.sqrt(loss), decimals=5),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        if self.image in ['image_s1', 'image_s2']:
            image, target = batch[self.image], batch['label']
        elif self.image == 'image_s1+s2':
            image = torch.concat([batch['image_s1'], batch['image_s2']], axis=1)
            target = batch['label']

        pred = self(image)
        loss = self.loss_fn(pred, target)

        self.log(
            "Val RMSE",
            torch.round(torch.sqrt(loss), decimals=5),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss


class _DoubleStreamUnet(pl.LightningModule):
    """
    Double Stream Unet as in: https://github.com/SebastianHafner/DS_UNet for 
    change detection using Sentinel-2 images
    """
    def __init__(self, dropout_p):
        """Expected input: [B, C, S, S] where B the batch size, C input channels and S the image length.
        The data values are expected to be standardized and [0, 1] scaled.
        
        - p: dropout probability
        """

        super().__init__()
        self.S1_Unet = _Unet(init_n_features=64, in_channels=4, dropout_p=dropout_p)
        self.S2_Unet = _Unet(init_n_features=64, in_channels=11, dropout_p=dropout_p)
        
        self.out_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0)

        return

    def forward(self, s1_image, s2_image):

        s1_feature = self.S1_Unet(s1_image)
        s2_feature = self.S2_Unet(s2_image)

        fusion = torch.cat((s1_feature, s2_feature), dim=1)

        out = self.out_conv(fusion)

        return out

class DoubleStreamUnet(_DoubleStreamUnet):
    def __init__(self, loss_fn, lr, dropout_p):
        """Expected input: [B, C, S, S] where B the batch size, C input channels and S the image length.
        The data values are expected to be Normalized.
        
        - p: dropout probability
        """
        super().__init__(dropout_p)
        self.lr = lr
        self.loss_fn = loss_fn

        self.save_hyperparameters()

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        factor = 0.1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    "min",
                    verbose=True,
                    patience=10,
                    min_lr=1e-8,
                    factor=factor,
                ),
                "monitor": "Val RMSE",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch, batch_idx):
        image_s1, image_s2, target = batch['image_s1'], batch['image_s2'], batch['label']

        pred = self(image_s1, image_s2)
        loss = self.loss_fn(pred, target)

        self.log(
            "Train RMSE",
            torch.round(torch.sqrt(loss), decimals=5),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        image_s1, image_s2, target = batch['image_s1'], batch['image_s2'], batch['label']

        pred = self(image_s1, image_s2)
        loss = self.loss_fn(pred, target)

        self.log(
            "Val RMSE",
            torch.round(torch.sqrt(loss), decimals=5),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss