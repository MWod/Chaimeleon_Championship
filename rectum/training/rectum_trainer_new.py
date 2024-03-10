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
from torchmetrics.classification import BinaryF1Score, BinaryAUROC

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
        self.f1_score_training_c1 = BinaryF1Score()
        self.f1_score_training_c2 = BinaryF1Score()
        self.f1_score_validation_c1 = BinaryF1Score()
        self.f1_score_validation_c2 = BinaryF1Score()
        self.auroc_training_c1 = BinaryAUROC()
        self.auroc_training_c2 = BinaryAUROC()
        self.auroc_validation_c1 = BinaryAUROC()
        self.auroc_validation_c2 = BinaryAUROC()
        self.conv_eval = training_params['conv_eval']
    
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
        if self.conv_eval:
            self.model.conv_layers.eval()
        input_data, features, ground_truth_c1, ground_truth_c2 = batch[0], batch[1], batch[2], batch[3]
        if self.use_features:
            output_c1, output_c2 = self.model(input_data[0], features)
        else:
            output_c1, output_c2 = self.model(input_data[0], None)
        if tc.any(tc.isnan(output_c1)) or tc.any(tc.isnan(output_c2)):
            print(f"NaN in Training output")
            return None
        loss_c1 = self.objective_function(output_c1, ground_truth_c1, **self.objective_function_params)
        loss_c2 = self.objective_function(output_c2, ground_truth_c2, **self.objective_function_params)
        loss = (loss_c1 + loss_c2) / 2.0
        one_hot_1 = utils.one_hot(ground_truth_c1, 2)
        one_hot_2 = utils.one_hot(ground_truth_c2, 2)
        f1_c1 = self.f1_score_training_c1(output_c1[0], one_hot_1[0])
        f1_c2 = self.f1_score_training_c2(output_c2[0], one_hot_2[0])
        f1 = (f1_c2 + f1_c2) / 2.0
        auroc_c1 = self.auroc_training_c1(output_c1[0], one_hot_1[0])
        auroc_c2 = self.auroc_training_c2(output_c2[0], one_hot_2[0])
        auroc = (auroc_c1 + auroc_c2) / 2.0
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/loss_c1", loss_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/loss_c2", loss_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/f1score_c1", f1_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/auroc_c1", auroc_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/f1score_c2", f1_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/auroc_c2", auroc_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss
                        
    def validation_step(self, batch, batch_idx):
        if self.conv_eval:
            self.model.conv_layers.eval()
        input_data, features, ground_truth_c1, ground_truth_c2 = batch[0], batch[1], batch[2], batch[3]
        if self.use_features:
            output_c1, output_c2 = self.model(input_data[0], features)
        else:
            output_c1, output_c2 = self.model(input_data[0], None)
        loss_c1 = self.objective_function(output_c1, ground_truth_c1, **self.objective_function_params)
        loss_c2 = self.objective_function(output_c2, ground_truth_c2, **self.objective_function_params)
        loss = (loss_c1 + loss_c2) / 2.0
        one_hot_1 = utils.one_hot(ground_truth_c1, 2)
        one_hot_2 = utils.one_hot(ground_truth_c2, 2)
        f1_c1 = self.f1_score_validation_c1(output_c1[0], one_hot_1[0])
        f1_c2 = self.f1_score_validation_c2(output_c2[0], one_hot_2[0])
        f1 = (f1_c2 + f1_c2) / 2.0
        auroc_c1 = self.auroc_validation_c1(output_c1[0], one_hot_1[0])
        auroc_c2 = self.auroc_validation_c2(output_c2[0], one_hot_2[0])
        auroc = (auroc_c1 + auroc_c2) / 2.0
        print(f"Output 1: {output_c1}")
        print(f"Output 2: {output_c2}")
        print(f"Ground truth 1: {ground_truth_c1}")
        print(f"Ground truth 2: {ground_truth_c2}")
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/loss_c1", loss_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/loss_c2", loss_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/f1score_c1", f1_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/auroc_c1", auroc_c1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/f1score_c2", f1_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/auroc_c2", auroc_c2, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

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

class RectumLightningTrainer():
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

