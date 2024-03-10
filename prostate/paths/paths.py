import pathlib

raw_data_path = pathlib.Path(r"/home/chaimeleon/datasets")
parsed_data_path = pathlib.Path(r"/home/chaimeleon/persistent-home/Dataset")


parsed_prostate_path = parsed_data_path / "Prostate"
prostate_csv_path = parsed_prostate_path / "dataset.csv"
training_prostate_csv_path = parsed_prostate_path / "training_dataset.csv"
validation_prostate_csv_path = parsed_prostate_path / "validation_dataset.csv"


project_path = pathlib.Path(r"/home/chaimeleon/persistent-home")
checkpoints_path = pathlib.Path(r"/home/chaimeleon/persistent-home/OpenChallenge/prostate")
logs_path = project_path / "Logs"
models_path = project_path / "models"
