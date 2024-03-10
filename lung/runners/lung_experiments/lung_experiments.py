### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib
import pickle

### External Imports ###
import numpy as np
import torch as tc
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv

### Internal Imports ###
from paths import paths as p
from input_output import volumetric as io_vol
from helpers import utils as u
from datasets import lung_dataset
from networks import milnet_lung
from augmentation import torchio as aug_tio

########################



def lung_experiment_2():
    training_csv_path = p.training_lung_csv_path
    validation_csv_path = p.validation_lung_csv_path
    # dataset_path = p.to_train_lung_path
    dataset_path = p.parsed_lung_path

    torchio_transforms = None
    loading_params = io_vol.default_volumetric_pytorch_load_params
    initial_thresholds = (-1000, 3000)
    scaler = 1
    load_mode = "event_1"

    transforms_path = p.models_path / "EffNetV2_S_Transforms"
    with open(transforms_path, 'rb') as handle:
        inference_transforms = pickle.load(handle)

    training_dataset = lung_dataset.LungDataset(dataset_path, training_csv_path, iteration_size=-1,
                                                torchio_transforms=torchio_transforms,
                                                loading_params=loading_params,
                                                initial_thresholds=initial_thresholds,
                                                inference_transforms=inference_transforms, scaler=scaler, load_mode=load_mode)
    validation_dataset = lung_dataset.LungDataset(dataset_path, validation_csv_path, iteration_size=-1,
                                                loading_params=loading_params,
                                                initial_thresholds=initial_thresholds,
                                                inference_transforms=inference_transforms, scaler=scaler, load_mode=load_mode)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")

    num_workers = 0
    training_dataloader = tc.utils.data.DataLoader(training_dataset, batch_size=1,
                                                        shuffle=True, num_workers=num_workers, pin_memory=False)
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                        shuffle=False, num_workers=num_workers, pin_memory=False)
    
    
    config = milnet_lung.config_lung_efficientnet()
    model = milnet_lung.MILNetExtended(**config)
    
    weights = tc.load(p.models_path / "EffNetV2_S_StateDict", map_location=tc.device('cpu'))
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend = tc.nn.Sequential(*list(backend.children())[:-1])
    backend.requires_grad_ = False
    
    ### Parameters ###
    experiment_name = "Chaimeleon_Lung_MILNetExtended_EffNetV2S_2"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    lr_decay = 0.98
    objective_function = lambda a, b: tc.mean((a-b)**2)
    objective_function_params = {}
    optimizer_weight_decay = 0.01
    use_features = True
    conv_eval = True

    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 8
    gradient_clip_val = 20
    reload_dataloaders_every_n_epochs = 1000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs
    
    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['model'] = model
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['conv_eval'] = conv_eval
    training_params['scaler'] = scaler
    training_params['backbone'] = backend

    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['use_features'] = use_features
    
    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params



