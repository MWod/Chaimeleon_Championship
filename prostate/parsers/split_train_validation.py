### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import pandas as pd


### Internal Imports ###
from paths import paths as p


########################


def prostate_split_train_val():
    split_ratio = 0.9
    seed = 1234

    input_csv_path = p.prostate_csv_path
    training_csv_path = p.training_prostate_csv_path
    validation_csv_path = p.validation_prostate_csv_path
    dataframe = pd.read_csv(input_csv_path)
    dataframe = dataframe.sample(frac=1, random_state=seed)
    training_dataframe = dataframe[:int(split_ratio*len(dataframe))]
    validation_dataframe = dataframe[int(split_ratio*len(dataframe)):]
    print(f"Dataset size: {len(dataframe)}")
    print(f"Training dataset size: {len(training_dataframe)}")
    print(f"Validation dataset size: {len(validation_dataframe)}")

    if not os.path.isdir(os.path.dirname(training_csv_path)):
        os.makedirs(os.path.dirname(training_csv_path))
    if not os.path.isdir(os.path.dirname(validation_csv_path)):
        os.makedirs(os.path.dirname(validation_csv_path))
    training_dataframe.to_csv(training_csv_path)
    validation_dataframe.to_csv(validation_csv_path)


def run():
    prostate_split_train_val()


if __name__ == "__main__":
    run()