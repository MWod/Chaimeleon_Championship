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
from torchmetrics.classification import BinaryF1Score, AUROC

### Internal Imports ###

########################

class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### General params
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
        self.f1_score_training = BinaryF1Score()
        self.f1_score_validation = BinaryF1Score()
        self.auroc_training = AUROC("binary")
        self.auroc_validation = AUROC("binary")
        self.conv_eval = training_params['conv_eval']
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = tc.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        dict = {'optimizer': optimizer}
        return dict
    
    def training_step(self, batch, batch_idx):
        if self.conv_eval:
            self.model.conv_layers.eval()
        input_data, features, ground_truth = batch[0], batch[1], batch[2]
        if self.use_features:
            output = self.model(input_data[0], features)
        else:
            output = self.model(input_data[0], None)
        if tc.any(tc.isnan(output)):
            print(f"NaN in Training output")
            return None
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        f1 = self.f1_score_training(output[0], utils.one_hot(ground_truth, 2)[0])
        auroc = self.auroc_training(output[0], utils.one_hot(ground_truth, 2)[0])
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss
                        
    def validation_step(self, batch, batch_idx):
        if self.conv_eval:
            self.model.conv_layers.eval()
        input_data, features, ground_truth = batch[0], batch[1], batch[2]
        if self.use_features:
            output = self.model(input_data[0], features)
        else:
            output = self.model(input_data[0], None)
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        f1 = self.f1_score_validation(output[0], utils.one_hot(ground_truth, 2)[0])
        auroc = self.auroc_validation(output[0], utils.one_hot(ground_truth, 2)[0])
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

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

class ProstateLightningTrainer():
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

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path) 
    
    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()

