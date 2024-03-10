import pathlib

raw_data_path = pathlib.Path(r"/home/chaimeleon/datasets")
parsed_data_path = pathlib.Path(r"/home/chaimeleon/persistent-home/Dataset")


parsed_rectum_path = parsed_data_path / "Rectum"
rectum_csv_path = parsed_rectum_path / "dataset.csv"
training_rectum_csv_path = parsed_rectum_path / "training_dataset.csv"
validation_rectum_csv_path = parsed_rectum_path / "validation_dataset.csv"


project_path = pathlib.Path(r"/home/chaimeleon/persistent-home")
checkpoints_path = pathlib.Path(r"/home/chaimeleon/persistent-home/OpenChallenge/rectum")
logs_path = project_path / "Logs"
models_path = project_path / "models"
