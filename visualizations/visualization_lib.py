import os

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator
from tensorflow import keras
from tensorflow.keras import backend as K

from interlacer import models
from scripts import training_config


def load_model(config_path):
    """

    Args:
      config_path(str): Path to model config file

    Returns:
      Keras model, without loaded weights

    """
    exp_config = training_config.TrainingConfig(config_path)
    exp_config.read_config()

    if(exp_config.dataset == 'FASTMRI'):
        n = 320
    elif(exp_config.dataset == 'MRI'):
        n = 256
    if(exp_config.architecture == 'CONV'):
        model = models.get_conv_no_residual_model(
            (n, n, 2), exp_config.nonlinearity, exp_config.kernel_size, exp_config.num_features, exp_config.num_layers)
    elif(exp_config.architecture == 'CONV_RESIDUAL'):
        model = models.get_conv_residual_model(
            (n, n, 2), exp_config.nonlinearity, exp_config.kernel_size, exp_config.num_features, exp_config.num_layers)
    elif(exp_config.architecture == 'INTERLACER_RESIDUAL'):
        model = models.get_interlacer_residual_model(
            (n, n, 2), exp_config.nonlinearity, exp_config.kernel_size, exp_config.num_features, exp_config.num_layers)
    elif(exp_config.architecture == 'ALTERNATING_RESIDUAL'):
        model = models.get_alternating_residual_model(
            (n, n, 2), exp_config.nonlinearity, exp_config.kernel_size, exp_config.num_features, exp_config.num_layers)

    return exp_config, model


def get_val_event_log(base_dir):
    """

    Args:
      base_dir: Path to model training directory

    Returns:
      (str) Path to validation tensorboard data

    """
    tb_str = os.path.join(base_dir,'tensorboard/validation/')
    return tb_str


def get_best_ckpt(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      (int) epoch with lowest loss

    """
    x = event_accumulator.EventAccumulator(
        path=get_val_event_log(model_path), size_guidance={
            event_accumulator.SCALARS: 0}, purge_orphaned_data=False)
    x.Reload()

    img_l1 = pd.DataFrame(x.Scalars('epoch_loss')).value[5::5]
        
    best_epoch = 5 * (np.argmin(img_l1) + 1) # epochs stored in intervals of 5
    return best_epoch
