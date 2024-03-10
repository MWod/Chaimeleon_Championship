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
from networks import double_milnet

########################


def rectum_inference_single(input_path, metadata_dict):
    ### Prepare Model ###
    device = "cuda:0" if tc.cuda.is_available() else "cpu"
    print(f"Cuda available: {tc.cuda.is_available()}")
    loading_params = v.default_volumetric_pytorch_load_params
    transforms_path = p.models_path / "EffNetV2_S_Transforms"
    with open(transforms_path, 'rb') as handle:
        inference_transforms = pickle.load(handle)

    weights = tc.load(p.models_path / "EffNetV2_S_StateDict", map_location=tc.device('cpu')) #TODO
    config = double_milnet.config_rectum_efficientnet(weights)
    model = double_milnet.MILNet(**config).to(device)

    # checkpoint_path = p.checkpoints_path / "Chaimeleon_Rectum_MIL_EffNetV2S_2" / "epoch=33_loss.ckpt"
    checkpoint_path = p.checkpoints_path / "Rectum.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    for key in all_keys:
        state_dict[key.replace("model.", "")] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(checkpoint['state_dict'])

    threshold_c1 = 0.40
    threshold_c2 = 0.38
    
    ### Parse Features ###
    age = metadata_dict['age']
    cea = metadata_dict['cea']
    no_previous_cancer = metadata_dict['no_previous_cancer']
    gender = metadata_dict['gender']
    features = np.zeros(4, dtype=np.float32) # Number of features in the JSON file
    ### Basic Features ###
    features[0] = age / 100.0 # normalized age
    features[1] = cea / 100.0 # normalized cea
    features[2] = no_previous_cancer
    features[3] = gender
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
    model.eval()
    with tc.no_grad():
        if input.shape[0] > 30:
            input = input[0:30]
        output_c1, output_c2 = model(input, features)

    risk_score_prob_c1 = tc.sigmoid(output_c1)[0, 1].item()
    risk_score_prob_c2 = tc.sigmoid(output_c2)[0, 1].item()

    extramural_vascular_invasion = risk_score_prob_c2 > threshold_c2
    extramural_vascular_invasion_prob = risk_score_prob_c2
    mesorectal_fascia_invasion = risk_score_prob_c1 > threshold_c1
    mesorectal_fascia_invasion_prob = risk_score_prob_c1

    extramural_vascular_invasion = 1 if extramural_vascular_invasion else 0
    mesorectal_fascia_invasion = 1 if mesorectal_fascia_invasion else 0
    return extramural_vascular_invasion, extramural_vascular_invasion_prob, mesorectal_fascia_invasion, mesorectal_fascia_invasion_prob

def run_rectum_evaluation(output_path):
    data_path = p.raw_data_path / "Rectum"

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
                if "harmonized" not in series['tags'][0]:
                    continue
                else:
                    break

            print(f"Input case path: {input_case_path}")
            ### Parse Metadata ###
            for i in range(len(metadatas)):
                subject_name = metadatas[i]['subjectName']
                if not (subject_name == study['subjectName']):
                    continue
                age = metadatas[i]['eForm']['pages'][0]['page_data']['age_at_diagnosis']['value']
                no_previous_cancer = metadatas[i]['eForm']['pages'][0]['page_data']['no_personal_history_cancer']['value']
                gender = metadatas[i]['eForm']['pages'][1]['page_data']['gender']['value']
                cea = metadatas[i]['eForm']['pages'][2]['page_data']['pret_cea_value']['value']

                if cea is None:
                    cea = 0
                gender_mapper = {'MALE': 0, 'FEMALE': 1}
                no_previous_cancer = 1 if no_previous_cancer else 0

                gender = gender_mapper[gender]
                cea = float(cea)
                metadata = {'age': age, 'no_previous_cancer': no_previous_cancer, 'gender': gender, 'cea': cea}

            print()
            print(f"Metadata: {metadata}")
            print()

            metadata_dict = metadata
            extramural_vascular_invasion, extramural_vascular_invasion_prob, mesorectal_fascia_invasion, mesorectal_fascia_invasion_prob = rectum_inference_single(input_case_path, metadata_dict)

            case = study['subjectName']
            to_append = (case, extramural_vascular_invasion, extramural_vascular_invasion_prob, mesorectal_fascia_invasion, mesorectal_fascia_invasion_prob)
            dataframe.append(to_append)

        except Exception as e:
            print(f"Excpetion: {e}")
            print(f"Error with the given case.")
            case = study['subjectName']
            extramural_vascular_invasion = 1
            extramural_vascular_invasion_prob = 0.5
            mesorectal_fascia_invasion = 1
            mesorectal_fascia_invasion_prob = 0.5
            to_append = (case, extramural_vascular_invasion, extramural_vascular_invasion_prob, mesorectal_fascia_invasion, mesorectal_fascia_invasion_prob)
            dataframe.append(to_append)
        
    dataframe = pd.DataFrame(dataframe, columns=['case', 'extramural_vascular_invasion', 'extramural_vascular_invasion_prob', 'mesorectal_fascia_invasion', 'mesorectal_fascia_invasion_prob'])
    dataframe.to_csv(output_path, index=False)

def run_evaluation_rectum_val_1():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "validation"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    output_path = "validation_rectum.csv"
    run_rectum_evaluation(output_path)

def run_evaluation_rectum_test_1():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "test"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    output_path = "test_rectum.csv"
    run_rectum_evaluation(output_path)


def run():
    # run_evaluation_rectum_val_1()
    run_evaluation_rectum_test_1()


if __name__ == "__main__":
    run()