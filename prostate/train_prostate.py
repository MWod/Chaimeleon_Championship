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
from training import prostate_trainer
from runners.prostate_experiments import prostate_experiments

########################


def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['lightning_params']['max_epochs']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, every_n_epochs=save_step, filename='{epoch}', save_top_k=-1)
    best_auroc_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_auroc', save_top_k=1, mode='max', monitor='Loss/Validation/auroc')
    best_f1_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_f1', save_top_k=1, mode='max', monitor='Loss/Validation/f1score')
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor='Loss/Validation/loss')
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
    training_params['lightning_params']['callbacks'] = [checkpoint_callback, best_auroc_checkpoint, best_f1_checkpoint, best_loss_checkpoint, best_training_loss_checkpoint] 
    training_params['checkpoints_path'] = checkpoints_path
    training_params['checkpoint_iters'] = checkpoints_iters
    training_params['log_image_iters'] = log_image_iters
    return training_params


def run_training(training_params):
    training_params = initialize(training_params)
    trainer = prostate_trainer.ProstateLightningTrainer(**training_params)
    trainer.run()

def prostate_training_1():
    run_training(prostate_experiments.prostate_experiment_2())


def run():
    prostate_training_1()


if __name__ == "__main__":
    run()