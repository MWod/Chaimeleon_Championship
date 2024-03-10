### Ecosystem Imports ###
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import pathlib

### External Imports ###
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

### Internal Imports ###
from paths import paths as p
from training import lung_trainer
from runners.lung_experiments import lung_experiments

########################


def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['lightning_params']['max_epochs']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, every_n_epochs=save_step, filename='{epoch}', save_top_k=-1)
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor='Loss/Validation/loss')
    real_mae_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_real_mae', save_top_k=1, mode='min', monitor='Loss/Validation/real_mae')
    real_mse_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_real_mse', save_top_k=1, mode='min', monitor='Loss/Validation/real_mse')
    best_training_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss_training', save_top_k=1, mode='min', monitor='Loss/Training/loss')
    checkpoints_iters = list(range(0, num_iterations, save_step))
    log_image_iters = list(range(0, num_iterations, save_step))
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    log_dir = os.path.join(p.logs_path, experiment_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [checkpoint_callback, best_loss_checkpoint, best_training_loss_checkpoint, real_mae_checkpoint, real_mse_checkpoint] 
    training_params['checkpoints_path'] = checkpoints_path
    training_params['checkpoint_iters'] = checkpoints_iters
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params):
    training_params = initialize(training_params)
    trainer = lung_trainer.LungLightningTrainerRegression(**training_params)
    trainer.run()

def lung_training_2():
    run_training(lung_experiments.lung_experiment_2())


def run():
    lung_training_2()
    pass


if __name__ == "__main__":
    run()