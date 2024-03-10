import sys
import os
import json
import shutil

import pandas as pd
import SimpleITK as sitk


def parse_prostate():
    input_data_path = r'/home/chaimeleon/datasets/Prostate'
    output_data_path = r'/home/chaimeleon/persistent-home/Dataset/Prostate'
    output_csv_path = os.path.join(output_data_path, "dataset.csv")
    echo = False

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
        try:
            # inner_paths = []
            for series_id, series in enumerate(study["series"]):
                ### Parse Image ###
                if echo:
                    print(f"Study: {study}")
                    print()
                    print(f"Series: {series}")
                input_case_path = os.path.join(input_data_path, study["path"], series["folderName"])
                if "harmonized" not in series['tags'][0]:
                    continue
                # inner_path = os.path.join(str(outer_id), f"{series['folderName']}_{series['tags'][0]}.nii.gz")
                # inner_paths.append(inner_path)
                inner_path = os.path.join(str(outer_id), f"harmonized.nii.gz")
                output_case_path = os.path.join(output_data_path, inner_path)
                if not os.path.exists(os.path.dirname(output_case_path)):
                    os.makedirs(os.path.dirname(output_case_path))

                print(f"Input case path: {input_case_path}")
                print(f"Output case path: {output_case_path}")

                # print(os.listdir(input_case_path))
                shutil.copy2(os.path.join(input_case_path, "harmonization_sample.nii.gz"), output_case_path)
                # image = sitk.ReadImage(os.path.join(input_case_path, "harmonization_sample.nii.gz"))
                
                # reader = sitk.ImageSeriesReader()
                # dicom_names = reader.GetGDCMSeriesFileNames(input_case_path)
                # reader.SetFileNames(dicom_names)
                # image = reader.Execute()
                # sitk.WriteImage(image, output_case_path)

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
                print(f"Ground truth: {ground_truth['risk_score']}")
                print()

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

            if echo:
                print()
                print(f"Metadata: {metadata}")
                print()

            to_append = (inner_path, str(ground_truth['risk_score']), float(metadata['age']), float(metadata['psa']), metadata['no_previous_cancer'])
            dataframe.append(to_append)
        except Exception as e:
            print(f"Excpetion: {e}")
            print(f"Error with the given case.")
        
    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth', 'Age', 'PSA', 'NoPreviousCancer'])
    dataframe.to_csv(output_csv_path, index=False)



def run():
    parse_prostate()

if __name__ == "__main__":
    run()