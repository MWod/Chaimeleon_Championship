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
from datasets import rectum_dataset_new
from networks import double_milnet
from augmentation import torchio as aug_tio

########################



def rectum_experiment_2():
    training_csv_path = p.training_rectum_csv_path
    validation_csv_path = p.validation_rectum_csv_path
    dataset_path = p.parsed_rectum_path

    torchio_transforms = aug_tio.final_transforms()
    loading_params = io_vol.default_volumetric_pytorch_load_params
    initial_thresholds = None

    transforms_path = p.models_path / "EffNetV2_S_Transforms"
    with open(transforms_path, 'rb') as handle:
        inference_transforms = pickle.load(handle)

    training_dataset = rectum_dataset_new.RectumDataset(dataset_path, training_csv_path, iteration_size=-1,
                                                torchio_transforms=torchio_transforms,
                                                loading_params=loading_params,
                                                initial_thresholds=initial_thresholds,
                                                inference_transforms=inference_transforms)
    validation_dataset = rectum_dataset_new.RectumDataset(dataset_path, validation_csv_path, iteration_size=-1,
                                                loading_params=loading_params,
                                                initial_thresholds=initial_thresholds,
                                                inference_transforms=inference_transforms)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")

    samples_weight = training_dataset.samples_weight
    print(f"Samples weight: {samples_weight}")
    print(f"Classes: {training_dataset.classes}")
    sampler = tc.utils.data.WeightedRandomSampler(tc.from_numpy(samples_weight).type('torch.DoubleTensor'), len(samples_weight))
    num_workers = 0
    outer_batch_size = 1
    training_dataloader = tc.utils.data.DataLoader(training_dataset, batch_size=1,
                                                        shuffle=False, num_workers=num_workers, pin_memory=False, sampler=sampler)
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                        shuffle=False, num_workers=num_workers, pin_memory=False)
    
    weights = tc.load(p.models_path / "EffNetV2_S_StateDict", map_location=tc.device('cpu'))
    config = double_milnet.config_rectum_efficientnet(weights)
    model = double_milnet.MILNet(**config)
    model.conv_layers.requires_grad_ = False
    
    ### Parameters ###
    experiment_name = "Chaimeleon_Rectum_MIL_EffNetV2S_2"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    lr_decay = 0.997
    objective_function = tc.nn.CrossEntropyLoss()
    objective_function_params = {}
    optimizer_weight_decay = 0.01
    use_features = True
    conv_eval = False

    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 201
    accumulate_grad_batches = 8
    gradient_clip_val = 100
    reload_dataloaders_every_n_epochs = 100000
    
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

    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['use_features'] = use_features
    
    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params




























