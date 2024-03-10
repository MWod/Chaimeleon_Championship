### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import time
import json
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###

from input_output import volumetric as v
from helpers import utils as u

########################


class LungDataset(tc.utils.data.Dataset):
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        iteration_size : int = -1,
        loading_params : dict = {},
        return_paths : bool=False,
        torchio_transforms = None,
        initial_thresholds=None,
        load_mode="all",
        inference_transforms=None,
        scaler=1,
        permute_images=False):
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)
        self.iteration_size = iteration_size
        self.torchio_transforms = torchio_transforms
        self.loading_params = loading_params
        self.return_paths = return_paths
        self.initial_thresholds=initial_thresholds
        self.inference_transforms = inference_transforms
        self.load_mode = load_mode
        self.scaler = scaler
        self.permute_images = permute_images
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)
            
        if self.load_mode == "all":
            pass
        elif self.load_mode == "event_1":
            events = []
            for idx in range(len(self.dataframe)):
                current_case = self.dataframe.loc[idx]
                ground_truth_event = current_case['Ground-Truth-Event']
                events.append(ground_truth_event)
            events = np.array(events)
            self.dataframe = self.dataframe[events == 1]
            self.dataframe.reset_index(drop=True, inplace=True)
        else:
            raise ValueError("Unsupported load mode.")

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def parse_metadata(self, metadata):
        age = metadata['Age']
        no_previous_cancer = metadata['NoPreviousCancer']
        gender = metadata['Gender']
        smoker = metadata['Smoker']
        packs_year = metadata['PacksYear']
        metastasis_lung = metadata['MetastasisLung']
        metastasis_adrenal_gland = metadata['MetastasisAdrenalGland']
        metastasis_liver = metadata['MetastasisLiver']
        metastasis_muscle = metadata['MetastasisMuscle']
        metastasis_brain = metadata['MetastasisBrain']
        metastasis_bone = metadata['MetastasisBone']
        metastasis_other = metadata['MetastasisOther']
        radiotherapy = metadata['Radiotherapy']
        chemotherapy = metadata['Chemotherapy']
        immunotherapy = metadata['Immunotherapy']
        surgery = metadata['Surgery']

        features = np.zeros(16, dtype=np.float32)
        ### Basic Features ###
        features[0] = age / 100.0 # normalized age
        features[1] = no_previous_cancer
        features[2] = gender
        features[3] = smoker
        features[4] = packs_year / 10.0
        features[5] = metastasis_lung
        features[6] = metastasis_adrenal_gland
        features[7] = metastasis_liver
        features[8] = metastasis_muscle
        features[9] = metastasis_brain
        features[10] = metastasis_bone
        features[11] = metastasis_other
        features[12] = radiotherapy
        features[13] = chemotherapy
        features[14] = immunotherapy
        features[15] = surgery
        ###
        return features
    

    def __getitem__(self, idx):
        current_case = self.dataframe.loc[idx]
        input_path = self.data_path / current_case['Input Path']
        ground_truth_months = current_case['Ground-Truth-Months'] / self.scaler
        features = self.parse_metadata(current_case)
        try:
            input_loader = v.VolumetricLoader(**self.loading_params).load(input_path)
            input, spacing, input_metadata = input_loader.volume, input_loader.spacing, input_loader.metadata
        except:
            return tc.tensor([1000])
        if self.initial_thresholds is not None:
            input[input < self.initial_thresholds[0]] = self.initial_thresholds[0]
            input[input > self.initial_thresholds[1]] = self.initial_thresholds[1]
            input = u.normalize_to_window(input, self.initial_thresholds[0], self.initial_thresholds[1])
        input = (input - tc.min(input)) / (tc.max(input) - tc.min(input))

        if self.torchio_transforms is not None:
            subject = tio.Subject(
            input = tio.ScalarImage(tensor=input))
            result = self.torchio_transforms(subject)
            transformed_input = result['input'].data
            transformed_input[0] = u.normalize(transformed_input[0])
            output = transformed_input  
        else:
            output = input

        output = output.unsqueeze(0)
        output = output.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        # output = output.permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        output = output.repeat(1, 3, 1, 1)
        output = self.inference_transforms(output)
        
        if self.permute_images:
            idx = tc.randperm(output.shape[0])
            output = output[idx]
        
        if self.return_paths:
            return output, features, ground_truth_months, spacing, dict(**current_case, **input_metadata)
        else:
            return output, features, ground_truth_months, spacing
        
        
        
        
        
        
        
        
        
        
