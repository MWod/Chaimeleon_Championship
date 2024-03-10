### Ecosystem Imports ###
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import pathlib
import json
import pickle

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd

### Internal Imports ###

from paths import paths as p
from input_output import volumetric as v
from helpers import utils as u
from networks import milnet

########################


def prostate_inference_single(input_path, metadata_dict):
    ### Prepare Model ###
    device = "cuda:0" if tc.cuda.is_available() else "cpu"
    print(f"Cuda available: {tc.cuda.is_available()}")
    loading_params = v.default_volumetric_pytorch_load_params
    transforms_path = p.models_path / "EffNetV2_S_Transforms"
    with open(transforms_path, 'rb') as handle:
        inference_transforms = pickle.load(handle)

    weights = tc.load(p.models_path / "EffNetV2_S_StateDict", map_location=tc.device('cpu'))
    config = milnet.config_prostate_efficientnet(weights)
    model = milnet.MILNet(**config).to(device)

    checkpoint_path = p.checkpoints_path / "Prostate.ckpt" #"Chaimeleon_Prostate_MIL_EffNetV2S_2" / "epoch=193_auroc.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    for key in all_keys:
        state_dict[key.replace("model.", "")] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(checkpoint['state_dict'])

    threshold = 0.38

    ### Parse Features ###
    age = metadata_dict['age']
    psa = metadata_dict['psa']
    no_previous_cancer = metadata_dict['no_previous_cancer']
    features = np.zeros(3, dtype=np.float32)
    features[0] = age / 100.0 # normalized age
    features[1] = psa / 100.0 # normalized psa
    features[2] = 0 if no_previous_cancer else 1
    features = tc.from_numpy(features).unsqueeze(0).to(device)

    ### Load & Preprocess Case ###
    input_loader = v.VolumetricLoader(**loading_params).load(input_path)
    input, spacing, input_metadata = input_loader.volume, input_loader.spacing, input_loader.metadata
    print(f"Loaded volume shape: {input.shape}")
    input = (input - tc.min(input)) / (tc.max(input) - tc.min(input))
    input = input.unsqueeze(0)
    print(f"Loaded volume shape: {input.shape}")
    input = input.permute(4, 1, 2, 3, 0)[:, :, :, :, 0].repeat(1, 3, 1, 1)
    input = inference_transforms(input)
    print(f"Loaded volume shape: {input.shape}")
    input = input.to(device)

    ### Run Inference And Return ###
    with tc.no_grad():
        output = model(input, features)
    risk_score_prob = tc.sigmoid(output)[0, 1].item()
    risk_score = risk_score_prob > threshold
    risk_score = 1 if risk_score else 0
    return risk_score, risk_score_prob

def run_prostate_evaluation(output_path):
    data_path = p.raw_data_path / "Prostate"

    ### Load General Files ###
    index_file_path = os.path.join(data_path, "index.json")
    with open(index_file_path) as file:
        studies = json.load(file)

    metadata_file_path = os.path.join(data_path, "eforms.json")
    with open(metadata_file_path) as file:
        metadatas = json.load(file)

    dataframe = []

    ### Run Inference For All Cases ###
    for outer_id, study in enumerate(studies):
        print(f"Outer ID: {outer_id} / {len(studies) - 1}")
        print(f"Study: {study}")
        try:
            for series_id, series in enumerate(study["series"]):
                ### Parse Image ###
                print(f"Study: {study}")
                print()
                print(f"Series: {series}")
                input_case_path = os.path.join(data_path, study["path"], series["folderName"], "harmonization_sample.nii.gz")
                if os.path.exists(input_case_path):
                    break

            print(f"Input case path: {input_case_path}")
            ### Parse Metadata ###
            for i in range(len(metadatas)):
                try:
                    subject_name = metadatas[i]['subjectName']
                    if not (subject_name == study['subjectName']):
                        continue
                    age = metadatas[i]['eForm']['pages'][0]['page_data']['age_at_diagnosis']['value']
                    no_previous_cancer = metadatas[i]['eForm']['pages'][0]['page_data']['no_personal_history_cancer']['value']
                    psa = metadatas[i]['eForm']['pages'][1]['page_data']['total_prostate_specific_antigen_level']['value']
                    metadata = {'age': age, 'no_previous_cancer': no_previous_cancer, 'psa': psa}
                except:
                    continue

            print()
            print(f"Metadata: {metadata}")
            print()

            metadata_dict = metadata
            risk_score, risk_score_prob = prostate_inference_single(input_case_path, metadata_dict)

            case = study['subjectName']
            to_append = (case, risk_score, risk_score_prob)
            dataframe.append(to_append)

        except Exception as e:
            print(f"Excpetion: {e}")
            print(f"Error with the given case.")
            case = study['subjectName']
            risk_score = 1
            risk_score_prob = 0.5
            to_append = (case, risk_score, risk_score_prob)
            dataframe.append(to_append)
        
    dataframe = pd.DataFrame(dataframe, columns=['case', 'risk_score', 'risk_score_prob'])
    dataframe.to_csv(output_path, index=False)

def run_evaluation_prostate_val_1():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "validation"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    output_path = "validation_prostate.csv"
    run_prostate_evaluation(output_path)

def run_evaluation_prostate_test_1():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "test"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    output_path = "test_prostate.csv"
    run_prostate_evaluation(output_path)


def run():
    # run_evaluation_prostate_val_1()
    run_evaluation_prostate_test_1()


if __name__ == "__main__":
    run()