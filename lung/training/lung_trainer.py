### Ecosystem Imports ###
import os
import sys

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Any, Callable
import pathlib
import time
import math

### External Imports ###
import numpy as np
import torch as tc
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import lightning as pl
from monai import metrics
from monai.networks import utils

### Internal Imports ###

########################


class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### General params
        self.backbone : tc.nn.Module = training_params['backbone']
        self.model : tc.nn.Module = training_params['model']
        self.learning_rate : float = training_params['learning_rate']
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.lr_decay : float = training_params['lr_decay']
        self.log_image_iters : Iterable[int]= training_params['log_image_iters']
        self.number_of_images_to_log : int = training_params['number_of_images_to_log']
        
        ## Cost functions and params
        self.objective_function : Callable = training_params['objective_function']
        self.objective_function_params : dict = training_params['objective_function_params']
        self.use_features = training_params['use_features']
        self.conv_eval = training_params['conv_eval']
        self.scaler = training_params['scaler']
        
        try:
            self.im_batch = training_params['im_batch']
        except:
            self.im_batch = None
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = tc.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: self.lr_decay ** epoch),
            "frequency": 1,
            "interval": "epoch",
        }
        dict = {'optimizer': optimizer, "lr_scheduler": scheduler}
        return dict
    
    def training_step(self, batch, batch_idx):
        test = batch[0]
        if len(test.shape) == 2 or len(test.shape) == 1:
            return None

        if self.conv_eval:
            self.backbone.eval()

        input_data, features, ground_truth = batch[0], batch[1], batch[2]
        if input_data is None:
            return None
        
        if self.use_features:
            with tc.no_grad():
                im_features = self.backbone(input_data[0])
            output = self.model(im_features, features)
        else:
            with tc.no_grad():
                im_features = self.backbone(input_data[0])
            output = self.model(im_features, None)
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        real_mse = (tc.mean((output * self.scaler - ground_truth * self.scaler)**2)).item()
        real_mae = (tc.mean(tc.abs((output * self.scaler - ground_truth * self.scaler)))).item()
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/real_mse", real_mse, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/real_mae", real_mae, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss
                        
    def validation_step(self, batch, batch_idx):
        test = batch[0]
        print(f"Len Shape: {len(test.shape)}")
        if len(test.shape) == 2 or len(test.shape) == 1:
            return None
        
        if self.conv_eval:
            self.backbone.eval()
        input_data, features, ground_truth = batch[0], batch[1], batch[2]
        if self.use_features:
            with tc.no_grad():
                im_features = self.backbone(input_data[0])
            output = self.model(im_features, features)
        else:
            with tc.no_grad():
                im_features = self.backbone(input_data[0])
            output = self.model(im_features, None)
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        real_mse = (tc.mean((output * self.scaler - ground_truth * self.scaler)**2)).item()
        real_mae = (tc.mean(tc.abs((output * self.scaler - ground_truth * self.scaler)))).item()
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/real_mse", real_mse, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/real_mae", real_mae, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        print()
        print(f"Input shape: {input_data.shape}")
        print(f"Ground truth: {ground_truth}")
        print(f"Output: {output}")

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, training_dataloader, validation_dataloader):
        super().__init__()
        self.td = training_dataloader
        self.vd = validation_dataloader
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.td.dataset.shuffle()
        return self.td
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.vd.dataset.shuffle()
        return self.vd

class LungLightningTrainerRegression():
    def __init__(self, **training_params : dict):
        ### General params
        self.training_dataloader : tc.utils.data.DataLoader = training_params['training_dataloader']
        self.validation_dataloader : tc.utils.data.DataLoader = training_params['validation_dataloader']
        lightning_params = training_params['lightning_params']    
        
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        if self.to_load_checkpoint_path is None:
            self.module = LightningModule(training_params, lightning_params)
        else:
            self.load_checkpoint()
            
        self.trainer = pl.Trainer(**lightning_params)
        self.data_module = LightningDataModule(self.training_dataloader, self.validation_dataloader)

    def save_checkpoint(self) -> None:
        self.trainer.save_checkpoint(pathlib.Path(self.checkpoints_path) / "Last_Iteration")
        tc.save(self.module.model.state_dict(), pathlib.Path(self.checkpoints_path) / "Direct_Save")

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path) 
    
    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()
