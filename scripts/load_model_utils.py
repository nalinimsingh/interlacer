import os

import numpy as np
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator

from scripts import training_config
from interlacer import models


def get_val_event_log(base_dir):
    """

    Args:
      base_dir: Path to model training directory

    Returns:
      (str) Path to validation tensorboard data

    """
    tb_str = os.path.join(base_dir, 'tensorboard/validation/')
    return tb_str


def get_train_event_log(base_dir):
    """

    Args:
      base_dir: Path to model training directory

    Returns:
      (str) Path to train tensorboard data

    """
    tb_str = os.path.join(base_dir, 'tensorboard/train/')
    return tb_str


def get_config_path(config_dir):
    """
    Args:
      config_dir(str): Path to dir containing model config file
    Returns:
      Path to config file
    """
    config_path = [i for i in os.listdir(config_dir) if i[-4:] == '.ini'][0]
    return config_path


def load_model(config_path):
    """
    Args:
      config_path(str): Path to model config file
    Returns:
      Keras model, without loaded weights
    """
    exp_config = training_config.TrainingConfig(config_path)
    exp_config.read_config()

    if('FASTMRI' in exp_config.dataset):
        n = None
    elif(exp_config.dataset == 'MRI'):
        n = 256
    if(exp_config.architecture == 'CONV'):
        model = models.get_conv_no_residual_model(
            (n,
             n,
             2),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_layers,
            exp_config.enforce_dc)
    elif(exp_config.architecture == 'CONV_RESIDUAL'):
        model = models.get_conv_residual_model(
            (n,
             n,
             2),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_layers,
            exp_config.enforce_dc)
    elif(exp_config.architecture == 'INTERLACER_RESIDUAL'):
        if('FASTMRI' in exp_config.dataset):
            model = models.get_fastmri_interlacer_residual_model(
                (None,
                 None,
                 2),
                exp_config.nonlinearity,
                exp_config.kernel_size,
                exp_config.num_features,
                exp_config.num_convs,
                exp_config.num_layers,
                exp_config.enforce_dc)
        else:
            model = models.get_interlacer_residual_model(
                (n,
                 n,
                 2),
                exp_config.nonlinearity,
                exp_config.kernel_size,
                exp_config.num_features,
                exp_config.num_convs,
                exp_config.num_layers,
                exp_config.enforce_dc)
    elif(exp_config.architecture == 'ALTERNATING_RESIDUAL'):
        model = models.get_alternating_residual_model(
            (n,
             n,
             2),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_layers,
            exp_config.enforce_dc)

    return exp_config, model


def get_best_ckpt(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      (int) epoch with lowest validation loss

    """
    x = event_accumulator.EventAccumulator(
        path=get_val_event_log(model_path), size_guidance={
            event_accumulator.SCALARS: 0}, purge_orphaned_data=False)
    x.Reload()

    img_l1 = pd.DataFrame(x.Scalars('epoch_loss')).value[10::10]

    best_epoch = 10 * (np.argmin(img_l1) + 1)  # epochs stored in intervals of 10
    return best_epoch


def get_last_ckpt(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      (int) index of last epoch

    """
    x = event_accumulator.EventAccumulator(
        path=get_val_event_log(model_path), size_guidance={
            event_accumulator.SCALARS: 0}, purge_orphaned_data=False)
    x.Reload()

    last_epoch = pd.DataFrame(x.Scalars('epoch_loss'))
    return last_epoch


def get_best_train_ckpt(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      (int) epoch with lowest train loss

    """
    x = event_accumulator.EventAccumulator(
        path=get_train_event_log(model_path), size_guidance={
            event_accumulator.SCALARS: 0}, purge_orphaned_data=False)
    x.Reload()

    losses = pd.DataFrame(x.Scalars('epoch_loss')).value[10::10]

    best_epoch = 10 * (np.argmin(losses) + 1)  # epochs stored in intervals of 10
    return best_epoch


def get_best_model(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      Keras model with loaded weights from best validation epoch

    """
    config_path = get_config_path(model_path)

    _, model = load_model(os.path.join(model_path, config_path))

    ckpt_num = get_best_ckpt(model_path)
    ckpt = str(ckpt_num).zfill(4)
    ckptname = 'cp-' + ckpt + '.' + 'ckpt'
    model.load_weights(os.path.join(model_path, ckptname)).expect_partial()

    return model
