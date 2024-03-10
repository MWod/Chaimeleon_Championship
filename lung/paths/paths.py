import pathlib

raw_data_path = pathlib.Path(r"/home/chaimeleon/datasets")
parsed_data_path = pathlib.Path(r"/home/chaimeleon/persistent-home/Dataset")


parsed_lung_path = parsed_data_path / "Lung"
to_train_lung_path = raw_data_path / "Lung"
lung_csv_path = parsed_lung_path / "dataset.csv"
training_lung_csv_path = parsed_lung_path / "training_dataset.csv"
validation_lung_csv_path = parsed_lung_path / "validation_dataset.csv"
    


project_path = pathlib.Path(r"/home/chaimeleon/persistent-home")
checkpoints_path = pathlib.Path(r"/home/chaimeleon/persistent-home/OpenChallenge/lung")
logs_path = project_path / "Logs"
models_path = project_path / "models"
