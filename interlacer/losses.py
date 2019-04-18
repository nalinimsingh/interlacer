import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from interlacer import utils


def fourier_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is computed on Fourier space real and imaginary data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input
    """
    if(output_domain == 'FREQ'):
        if(loss == 'L1'):
            def fourier_l1(y_true, y_pred):
                return K.mean(K.abs(y_true - y_pred))
            return fourier_l1
        elif(loss == 'L2'):
            def fourier_l2(y_true, y_pred):
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return fourier_l2
    elif(output_domain == 'IMAGE'):
        if(loss == 'L1'):
            def fourier_l1(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(
                    y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(
                    y_pred)
                return K.mean(K.abs(y_true_fourier - y_pred_fourier))
            return fourier_l1
        elif(loss == 'L2'):
            def fourier_l2(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(
                    y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(
                    y_pred)
                return K.mean(K.pow(K.abs(y_true_fourier - y_pred_fourier), 2))
            return fourier_l2


def image_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is computed on image space real and imaginary data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input
    """
    if(output_domain == 'IMAGE'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                return K.mean(K.abs(y_true - y_pred))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return image_l2
    elif(output_domain == 'FREQ'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_true, axes=(1, 2)))
                y_pred_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_pred, axes=(1, 2)))
                return K.mean(K.abs(y_true_fourier - y_pred_fourier))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_true, axes=(1, 2)))
                y_pred_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_pred, axes=(1, 2)))
                return K.mean(K.pow(K.abs(y_true_fourier - y_pred_fourier), 2))
            return image_l2


def image_mag_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is computed on image space magnitude data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input
    """
    if(output_domain == 'IMAGE'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                return K.mean(
                    K.abs(
                        K.abs(
                            utils.join_reim_tensor(y_true)) -
                        K.abs(
                            utils.join_reim_tensor(y_pred))))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return image_l2
    elif(output_domain == 'FREQ'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_true, axes=(1, 2)))
                y_pred_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_pred, axes=(1, 2)))
                return K.mean(
                    K.abs(
                        K.abs(
                            utils.join_reim_tensor(y_true_fourier)) -
                        K.abs(
                            utils.join_reim_tensor(y_pred_fourier))))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_true, axes=(1, 2)))
                y_pred_fourier = utils.convert_tensor_to_image_domain(
                    tf.signal.ifftshift(y_pred, axes=(1, 2)))
                return K.mean(K.pow(K.abs(y_true_fourier - y_pred_fourier), 2))
            return image_l2
