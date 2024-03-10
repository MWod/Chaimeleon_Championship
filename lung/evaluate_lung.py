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
import SimpleITK as sitk
import torchvision as tv

### Internal Imports ###

from paths import paths as p
from input_output import volumetric as v
from helpers import utils as u
from networks import milnet_lung
from preprocessing import preprocessing_volumetric as pre_vol

########################


def lung_inference_single(volume, metadata_dict, checkpoint_path):
    ### Prepare Model ###
    device = "cuda:0" if tc.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Cuda available: {tc.cuda.is_available()}")
    transforms_path = p.models_path / "EffNetV2_S_Transforms"
    with open(transforms_path, 'rb') as handle:
        inference_transforms = pickle.load(handle)
        
    scaler = 1
    initial_thresholds = (-1000, 3000)
    
    weights = tc.load(p.models_path / "EffNetV2_S_StateDict", map_location=tc.device('cpu'))
    backend = tv.models.efficientnet_v2_s(weights=None)
    backend.load_state_dict(weights)
    backend = tc.nn.Sequential(*list(backend.children())[:-1])
    backend.requires_grad_ = False
    backend = backend.to(device)
    
    config = milnet_lung.config_lung_efficientnet()
    model = milnet_lung.MILNetExtended(**config).to(device)
    model.eval()
         
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    for key in all_keys:
        state_dict[key.replace("model.", "")] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(checkpoint['state_dict'])
    
    backend.eval()
    model.eval()

    ### Parse Features ###
    age = metadata_dict['age']
    no_previous_cancer = metadata_dict['no_previous_cancer']
    gender = metadata_dict['gender']
    smoker = metadata_dict['smoker']
    packs_year = metadata_dict['packs_year']
    metastasis_lung = metadata_dict['metastasis_lung']
    metastasis_adrenal_gland = metadata_dict['metastasis_adrenal_gland']
    metastasis_liver = metadata_dict['metastasis_liver']
    metastasis_muscle = metadata_dict['metastasis_muscle']
    metastasis_brain = metadata_dict['metastasis_brain']
    metastasis_bone = metadata_dict['metastasis_bone']
    metastasis_other = metadata_dict['metastasis_other']
    radiotherapy = metadata_dict['radiotherapy']
    chemotherapy = metadata_dict['chemotherapy']
    immunotherapy = metadata_dict['immunotherapy']
    surgery = metadata_dict['surgery']
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
    features = tc.from_numpy(features).unsqueeze(0).to(device)

    ### Load & Preprocess Case ###
    input = tc.from_numpy(volume).unsqueeze(0)
    print(f"Loaded volume shape: {input.shape}")
    if initial_thresholds is not None:
        input[input < initial_thresholds[0]] = initial_thresholds[0]
        input[input > initial_thresholds[1]] = initial_thresholds[1]
        input = u.normalize_to_window(input, initial_thresholds[0], initial_thresholds[1])
    input = (input - tc.min(input)) / (tc.max(input) - tc.min(input))
    print(f"Loaded volume shape: {input.shape}")
    input = input.unsqueeze(0)
    input = input.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    input = input.repeat(1, 3, 1, 1)
    input = inference_transforms(input)
    print(f"Loaded volume shape: {input.shape}")
    input = input.to(device)

    ### Run Inference And Return ###
    with tc.no_grad():
        im_features = backend(input)
        output = model(im_features, features)
        output = tc.abs(output) * scaler
            
    predicted_months = output.cpu().item()
    return predicted_months


def parse_case(volume_path, output_size, device="cpu"):
    volume = sitk.ReadImage(volume_path)
    spacing = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")
    volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Volume TC shape: {volume_tc.shape}")
    resampled_volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
    print(f"Resampled Volume TC shape: {resampled_volume_tc.shape}")
    resampled_volume_tc = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy()
    new_spacing = tuple(np.array(spacing) * np.array(volume.shape) / np.array(output_size))
    return resampled_volume_tc, spacing, new_spacing

def run_lung_evaluation(output_path, checkpoint_path):
    data_path = p.raw_data_path / "Lung"
    device = "cuda:0" if tc.cuda.is_available() else "cpu"

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
                print(f"Input case path: {input_case_path}")

                output_size = (128, 384, 384)
                volume, spacing, new_spacing = parse_case(input_case_path, output_size, device=device)
                print(f"Spacing: {spacing}")
                print(f"New Spacing: {new_spacing}")
                print(f"Resampled shape: {volume.shape}")
                
            ### Parse Metadata ###
            for i in range(len(metadatas)):
                subject_name = metadatas[i]['subjectName']
                if not (subject_name == study['subjectName']):
                    continue

                age = metadatas[i]['eForm']['pages'][0]['page_data']['age_at_baseline']['value']
                no_previous_cancer = metadatas[i]['eForm']['pages'][0]['page_data']['no_personal_history_cancer']['value']
                gender = metadatas[i]['eForm']['pages'][1]['page_data']['gender']['value']
                smoker = metadatas[i]['eForm']['pages'][2]['page_data']['smoking_status']['value']
                packs_year = metadatas[i]['eForm']['pages'][2]['page_data']['packs_year']['value']
                packs_year = 0 if packs_year is None else packs_year

                no_previous_cancer = 1 if no_previous_cancer else 0
                gender = 1 if gender == 'MALE' else 0

                smoker_mapper = {'Non-smoker': 1, 'Ex-smoker': 2, 'Smoker': 3, 'Unknown': 0, None: 0}
                smoker = smoker_mapper[smoker]

                mapper_metastasis = {False: 0, True: 1}
                metastasis_lung = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_lung']['value']
                metastasis_adrenal_gland = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_adrenal_gland']['value']
                metastasis_liver = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_liver']['value']
                metastasis_muscle = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_muscle']['value']
                metastasis_brain = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_brain']['value']
                metastasis_bone = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_bone']['value']
                metastasis_other = metadatas[i]['eForm']['pages'][2]['page_data']['metastasis_other']['value']

                metastasis_lung = mapper_metastasis[metastasis_lung]
                metastasis_adrenal_gland = mapper_metastasis[metastasis_adrenal_gland]
                metastasis_liver = mapper_metastasis[metastasis_liver]
                metastasis_muscle = mapper_metastasis[metastasis_muscle]
                metastasis_brain = mapper_metastasis[metastasis_brain]
                metastasis_bone = mapper_metastasis[metastasis_bone]
                metastasis_other = mapper_metastasis[metastasis_other]

                mapper_therapy = {None: 0, 'No': 1, 'Yes': 2, False: 1, True: 2, 'Unknown': 0}
                radiotherapy = metadatas[i]['eForm']['pages'][3]['page_data']['radiotherapy']['value']
                chemotherapy = metadatas[i]['eForm']['pages'][3]['page_data']['chemotherapy']['value']
                immunotherapy = metadatas[i]['eForm']['pages'][3]['page_data']['immunotherapy']['value']
                surgery = metadatas[i]['eForm']['pages'][3]['page_data']['surgery']['value']

                radiotherapy = mapper_therapy[radiotherapy]
                chemotherapy = mapper_therapy[chemotherapy]
                immunotherapy = mapper_therapy[immunotherapy]
                surgery = mapper_therapy[surgery]

                metadata = {'age': age, 'no_previous_cancer': no_previous_cancer, 'gender': gender, 'smoker': smoker, 'packs_year': packs_year,
                            'metastasis_lung' : metastasis_lung, 'metastasis_adrenal_gland': metastasis_adrenal_gland, 'metastasis_liver' : metastasis_liver,
                            'metastasis_muscle' : metastasis_muscle, 'metastasis_brain' : metastasis_brain, 'metastasis_bone' : metastasis_bone,
                            'metastasis_other' : metastasis_other, 'radiotherapy' : radiotherapy, 'chemotherapy' : chemotherapy, 
                            'immunotherapy': immunotherapy, 'surgery': surgery}

            print()
            print(f"Metadata: {metadata}")
            print()
            
            metadata_dict = metadata
            survival_time_months = lung_inference_single(volume, metadata_dict, checkpoint_path)
            print(f"Time: {survival_time_months}")
            case = study['subjectName']
            to_append = (case, survival_time_months)
            dataframe.append(to_append)

        except Exception as e:
            print(f"Excpetion: {e}")
            print(f"Error with the given case.")
            case = study['subjectName']
            survival_time_months = 10.0
            to_append = (case, survival_time_months)
            dataframe.append(to_append)
        
    dataframe = pd.DataFrame(dataframe, columns=['case', 'survival_time_months'])
    dataframe.to_csv(output_path, index=False)


def run_evaluation_lung_val():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "validation"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    output_path = "validation_lung.csv"
    checkpoint_path = p.checkpoints_path / "Lung.ckpt" #"Chaimeleon_Lung_MILNetExtended_EffNetV2S_2" / "epoch=43_real_mae.ckpt"
    run_lung_evaluation(output_path, checkpoint_path)
    
def run_evaluation_lung_test():
    """
    jobman submit -i ubuntu-python -a '{"chaimeleon.eu/openchallengeJob": "test"}' -- python3 ~/persistent-home/src/evaluate.py
    """
    checkpoint_path = p.checkpoints_path / "Lung" #"Chaimeleon_Lung_MILNetExtended_EffNetV2S_2" / "epoch=43_real_mae.ckpt"
    output_path = "test_lung.csv"
    run_lung_evaluation(output_path, checkpoint_path)


def run():
    # run_evaluation_lung_val()
    run_evaluation_lung_test()


if __name__ == "__main__":
    run()