import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import layers, utils


def get_conv_no_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_layers):
    """Generic conv model without residual convolutions.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_layers(int): Number of convolutional layers in model

    Returns:
      model: Keras model comprised of num_layers core convolutional layers with specified nonlinearities

    """
    inputs = Input(input_size)
    prev_layer = inputs
    for i in range(num_layers):
        conv = layers.BatchNormConv(num_features, kernel_size)(prev_layer)
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv)
        prev_layer = nonlinear
    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer)
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_conv_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_layers):
    """Generic conv model with residual convolutions.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_layers(int): Number of convolutional layers in model

    Returns:
      model: Keras model comprised of num_layers core convolutional layers with specified nonlinearities

    """
    inputs = Input(input_size)

    prev_layer = inputs
    for i in range(num_layers):
        conv = layers.BatchNormConv(num_features, kernel_size)(prev_layer)
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv)
        prev_layer = nonlinear + tf.tile(inputs, [1, 1, 1, int(num_features/2)])
    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer) + inputs
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_interlacer_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_layers):
    """Model with residual convolutions.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_layers(int): Number of convolutional layers in model

    Returns:
      model: Keras model comprised of num_layers core interlaced layers with specified nonlinearities

    """
    inputs = Input(input_size)
    n = inputs.get_shape().as_list()[1]
    inp_real = tf.expand_dims(inputs[:, :, :, 0], -1)
    inp_imag = tf.expand_dims(inputs[:, :, :, 1], -1)

    n_copies = int(num_features / 2)

    inp_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_real, inp_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    complex_inputs = Permute((3, 1, 2))(utils.join_reim_channels(inputs))
    inputs_img = utils.split_reim_channels(
        Permute(
            (2, 3, 1))(
            tf.signal.ifft2d(complex_inputs)))
    inp_img_real = tf.expand_dims(inputs_img[:, :, :, 0], -1)
    inp_img_imag = tf.expand_dims(inputs_img[:, :, :, 1], -1)

    inp_img_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_img_real, inp_img_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    freq_in = inputs
    img_in = inputs_img

    for i in range(num_layers):
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size)([img_in, freq_in])

        freq_in = k_conv + inp_copy
        img_in = img_conv + inp_img_copy

    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(freq_in) + inputs
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model
