### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk


### Internal Imports ###
from paths import paths as p
from input_output import volumetric as v
from helpers import utils as u
from input_output import volumetric as io_vol
from preprocessing import preprocessing_volumetric as pre_vol


########################


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

def parse_lung():
    input_data_path = r'/home/chaimeleon/datasets/Lung'
    output_data_path = p.parsed_lung_path_2
    output_csv_path = os.path.join(output_data_path, "dataset.csv")
    echo = True

    dataframe = []

    ### Load General Files ###
    index_file_path = os.path.join(input_data_path, "index.json")
    with open(index_file_path) as file:
        studies = json.load(file)

    ground_truth_file_path = os.path.join(input_data_path, "ground_truth.json")
    with open(ground_truth_file_path) as file:
        ground_truths = json.load(file)

    metadata_file_path = os.path.join(input_data_path, "eforms.json")
    with open(metadata_file_path) as file:
        metadatas = json.load(file)

    if echo:
        print()
        print(type(ground_truths))
        print(ground_truths[0])
        print()

        print()
        print(type(metadatas))
        print(metadatas[0])
        print()

    ### Perform Parsing ###
    for outer_id, study in enumerate(studies):
        print(f"Outer ID: {outer_id} / {len(studies) - 1}")
        # try:
            # inner_paths = []
        try:
            for series_id, series in enumerate(study["series"]):
                ### Parse Image ###
                if echo:
                    print(f"Study: {study}")
                    print()
                    print(f"Series: {series}")
                input_case_path = os.path.join(input_data_path, study["path"], series["folderName"])
                if "harmonized" not in series['tags'][0]:
                    continue
                # inner_path = os.path.join(str(outer_id), f"harmonized.nii.gz")
                inner_path = os.path.join(study["path"], series["folderName"], "harmonization_sample.nii.gz")
                output_case_path = os.path.join(output_data_path, inner_path)
                # if not os.path.exists(os.path.dirname(output_case_path)):
                #     os.makedirs(os.path.dirname(output_case_path))
                print(f"Input case path: {input_case_path}")
                print(f"Output case path: {output_case_path}")
                print(f"Inner path: {inner_path}")
                # shutil.copy2(os.path.join(input_case_path, "harmonization_sample.nii.gz"), output_case_path)

                output_size = (128, 384, 384)
                volume, spacing, new_spacing = parse_case(os.path.join(input_data_path, inner_path), output_size, "cpu")
                print(f"Spacing: {spacing}")
                print(f"New Spacing: {new_spacing}")
                print(f"Resampled shape: {volume.shape}")
                to_save = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0))
                to_save.SetSpacing(new_spacing)
                if not os.path.exists(os.path.dirname(output_case_path)):
                    os.makedirs(os.path.dirname(output_case_path))
                sitk.WriteImage(to_save, str(output_case_path))

        except Exception as e:
            print(e)
            print("Error during loading the file.")
                
        ### Parse Ground-Truth ###
        for i in range(len(ground_truths)):
            try:
                subject_name = ground_truths[i]['subjectName']
                if not (subject_name == study['subjectName']):
                    continue
                ground_truth = ground_truths[i]['groundTruth']
            except:
                continue

        if echo:
            print()
            print(f"Ground truth survival: {ground_truth['survival_time_months']}")
            print(f"Ground truth event: {ground_truth['event']}")
            print()

        ### Parse Metadata ###
        for i in range(len(metadatas)):
            # try:
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

        if echo:
            print()
            print(f"Metadata: {metadata}")
            print()

        to_append = (inner_path, float(ground_truth['survival_time_months']), float(ground_truth['event']),
                        float(metadata['age']), metadata['no_previous_cancer'], metadata['gender'], metadata['smoker'],
                        float(metadata['packs_year']), metadata['metastasis_lung'], metadata['metastasis_adrenal_gland'],
                        metadata['metastasis_liver'], metadata['metastasis_muscle'], metadata['metastasis_brain'],
                        metadata['metastasis_bone'], metadata['metastasis_other'], metadata['radiotherapy'],
                        metadata['chemotherapy'], metadata['immunotherapy'], metadata['surgery'])
        dataframe.append(to_append)
        
    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth-Months', 'Ground-Truth-Event',
                                                 'Age', 'NoPreviousCancer', 'Gender', 'Smoker', 'PacksYear',
                                                 'MetastasisLung', 'MetastasisAdrenalGland', 'MetastasisLiver',
                                                 'MetastasisMuscle', 'MetastasisBrain', 'MetastasisBone',
                                                 'MetastasisOther', 'Radiotherapy', 'Chemotherapy',
                                                 'Immunotherapy', 'Surgery'])
    dataframe.to_csv(output_csv_path, index=False)


def run():
    parse_lung()

if __name__ == "__main__":
    run()