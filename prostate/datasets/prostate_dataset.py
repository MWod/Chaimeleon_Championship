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


class ProstateDataset(tc.utils.data.Dataset):
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        iteration_size : int = -1,
        loading_params : dict = {},
        return_paths : bool=False,
        torchio_transforms = None,
        initial_thresholds=None,
        inference_transforms=None):
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
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)

        class_0_counter = 0
        class_1_counter = 1
        for i in range(len(self.dataframe)):
            current_case = self.dataframe.loc[i]
            gt = current_case['Ground-Truth']
            current_class = 0 if gt == "Low" else 1
            if current_class == 0:
                class_0_counter += 1
            else:
                class_1_counter += 1
        
        samples_weight = []
        classes = []
        for i in range(len(self.dataframe)):
            current_case = self.dataframe.loc[i]
            gt = current_case['Ground-Truth']
            current_class = 0 if gt == "Low" else 1
            if current_class == 0:
                samples_weight.append(class_0_counter / (class_0_counter + class_1_counter))
                classes.append(0)
            else:
                samples_weight.append(class_1_counter / (class_0_counter + class_1_counter))
                classes.append(1)

        self.samples_weight = 1 / np.array(samples_weight)
        self.classes = np.array(classes)

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def parse_metadata(self, current_case):
        age = current_case['Age']
        psa = current_case['PSA']
        no_previous_cancer = current_case['NoPreviousCancer']

        features = np.zeros(3, dtype=np.float32) # Number of features in the JSON file
        ### Basic Features ###
        features[0] = age / 100.0 # normalized age
        features[1] = psa / 100.0 # normalized psa
        features[2] = 0 if no_previous_cancer else 1
        ###
        return features

    def __getitem__(self, idx):
        current_case = self.dataframe.loc[idx]
        input_path = self.data_path / current_case['Input Path']
        ground_truth = current_case['Ground-Truth']
        ground_truth = 0 if ground_truth == "Low" else 1
        features = self.parse_metadata(current_case)
        input_loader = v.VolumetricLoader(**self.loading_params).load(input_path)
        input, spacing, input_metadata = input_loader.volume, input_loader.spacing, input_loader.metadata
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
        output = output.permute(4, 1, 2, 3, 0)[:, :, :, :, 0].repeat(1, 3, 1, 1)
        output = self.inference_transforms(output)
        if self.return_paths:
            return output, features, ground_truth, spacing, dict(**current_case, **input_metadata)
        else:
            return output, features, ground_truth, spacing