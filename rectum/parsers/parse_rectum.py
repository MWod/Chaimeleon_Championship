### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import shutil

### External Imports ###
import numpy as np
import pandas as pd
import SimpleITK as sitk


### Internal Imports ###
from paths import paths as p


########################


def parse_rectum():
    input_data_path = r'/home/chaimeleon/datasets/Rectum'
    output_data_path = p.parsed_rectum_path
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

    counter_class_1 = 0
    counter_class_2 = 0
    ### Perform Parsing ###
    for outer_id, study in enumerate(studies):
        print(f"Outer ID: {outer_id} / {len(studies) - 1}")
        for series_id, series in enumerate(study["series"]):
            ### Parse Image ###
            if echo:
                print(f"Study: {study}")
                print()
                print(f"Series: {series}")
            input_case_path = os.path.join(input_data_path, study["path"], series["folderName"])
            if "harmonized" not in series['tags'][0]:
                continue
            inner_path = os.path.join(str(outer_id), f"harmonized.nii.gz")
            output_case_path = os.path.join(output_data_path, inner_path)
            if not os.path.exists(os.path.dirname(output_case_path)):
                os.makedirs(os.path.dirname(output_case_path))

            print(f"Input case path: {input_case_path}")
            print(f"Output case path: {output_case_path}")

            shutil.copy2(os.path.join(input_case_path, "harmonization_sample.nii.gz"), output_case_path)

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
            print(f"Ground truth 1: {ground_truth['mesorectal_invation']}")
            print(f"Ground truth 2: {ground_truth['extramural_invation']}")
            if ground_truth['mesorectal_invation']:
                counter_class_1 += 1
            if ground_truth['extramural_invation']:
                counter_class_2 += 1
            print()

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

        if echo:
            print()
            print(f"Metadata: {metadata}")
            print()

        ground_truth_1 = 1 if ground_truth['mesorectal_invation'] else 0
        ground_truth_2 = 1 if ground_truth['extramural_invation'] else 0
        to_append = (inner_path, ground_truth_1, ground_truth_2,
                     float(metadata['age']), float(metadata['cea']), metadata['no_previous_cancer'], metadata['gender'])
        dataframe.append(to_append)
    print(f"Counter 1: {counter_class_1}")
    print(f"Counter 2: {counter_class_2}")
        
    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth-Mesorectal', 'Ground-Truth-Extramural',
                                                  'Age', 'CEA', 'NoPreviousCancer', 'Gender'])
    dataframe.to_csv(output_csv_path, index=False)


def run():
    parse_rectum()

if __name__ == "__main__":
    run()