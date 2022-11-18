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
        num_layers,
        enforce_dc):
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
    if(enforce_dc):
        masks = Input(input_size)

    prev_layer = inputs
    for i in range(num_layers):
        conv = layers.BatchNormConv(num_features, kernel_size)(prev_layer)
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv)
        prev_layer = nonlinear
    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer)

    if(enforce_dc):
        output = masks * inputs + (1 - masks) * output
        model = keras.models.Model(inputs=(inputs, masks), outputs=output)
    else:
        model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_conv_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_layers,
        enforce_dc):
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
    if(enforce_dc):
        masks = Input(input_size)

    prev_layer = inputs
    for i in range(num_layers):
        conv = layers.BatchNormConv(num_features, kernel_size)(prev_layer)
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv)
        prev_layer = nonlinear + \
            tf.tile(inputs, [1, 1, 1, int(num_features / 2)])
    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer) + inputs

    if(enforce_dc):
        output = masks * inputs + (1 - masks) * output
        model = keras.models.Model(inputs=(inputs, masks), outputs=output)
    else:
        model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_interlacer_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        enforce_dc):
    """Interlacer model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns a frequency-space output of the same size, comprised of interlacer layers and with connections from the input to each layer.

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
    if(enforce_dc):
        masks = Input(input_size)

    n = inputs.get_shape().as_list()[1]
    inp_real = tf.expand_dims(inputs[:, :, :, 0], -1)
    inp_imag = tf.expand_dims(inputs[:, :, :, 1], -1)

    n_copies = int(num_features / 2)

    inp_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_real, inp_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    inputs_img = utils.convert_tensor_to_image_domain(inputs)
    inp_img_real = tf.expand_dims(inputs_img[:, :, :, 0], -1)
    inp_img_imag = tf.expand_dims(inputs_img[:, :, :, 1], -1)

    inp_img_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_img_real, inp_img_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    freq_in = inputs
    img_in = inputs_img

    for i in range(num_layers):
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs)([img_in, freq_in])

        freq_in = k_conv + inp_copy
        img_in = img_conv + inp_img_copy

    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(freq_in) + inputs

    if(enforce_dc):
        output = masks * inputs + (1 - masks) * output
        model = keras.models.Model(inputs=(inputs, masks), outputs=output)
    else:
        model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def crop_320(inputs):
    inputs = tf.expand_dims(inputs, 0)
    inputs_img = utils.convert_tensor_to_image_domain(inputs)[0, :, :, :]
    inputs_img = tf.signal.ifftshift(inputs_img, axes=(0, 1))
    shape = tf.shape(inputs_img)
    x = shape[0]
    y = shape[1]
    n = 320

    x_l = tf.cast(x / 2 - n / 2, tf.int32)
    x_r = tf.cast(x / 2 + n / 2, tf.int32)
    y_l = tf.cast(y / 2 - n / 2, tf.int32)
    y_r = tf.cast(y / 2 + n / 2, tf.int32)

    icrop_img = tf.expand_dims(
        tf.slice(inputs_img, (x_l, y_l, 0), (n, n, 2)), 0)
    icrop_k = utils.convert_tensor_to_frequency_domain(icrop_img)[0, :, :, :]
    return icrop_k


def get_fastmri_interlacer_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        enforce_dc):
    """Interlacer model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns
    a frequency-space output of the same size, comprised of interlacer layers and with connections
    from the input to each layer. Handles variable input size, and crops to a 320x320 image at the end.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_convs(int): Number of convolutions per layer
      num_layers(int): Number of convolutional layers in model
      enforce_dc(Bool): Whether to paste in original acquired k-space lines in final output

    Returns:
      model: Keras model comprised of num_layers core interlaced layers with specified nonlinearities

    """
    inputs = Input(input_size)
    if(enforce_dc):
        masks = Input(input_size)

    x = tf.shape(inputs)[1]
    y = tf.shape(inputs)[2]
    inp_real = tf.expand_dims(inputs[:, :, :, 0], -1)
    inp_imag = tf.expand_dims(inputs[:, :, :, 1], -1)

    n_copies = int(num_features / 2)

    inp_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_real, inp_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, x, y, num_features])

    inputs_img = utils.convert_tensor_to_image_domain(inputs)
    inp_img_real = tf.expand_dims(inputs_img[:, :, :, 0], -1)
    inp_img_imag = tf.expand_dims(inputs_img[:, :, :, 1], -1)

    inp_img_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_img_real, inp_img_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, x, y, num_features])

    freq_in = inputs
    img_in = inputs_img

    for i in range(num_layers):
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs, shift=True)([img_in, freq_in])

        freq_in = k_conv + inp_copy
        img_in = img_conv + inp_img_copy

    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(freq_in) + inputs

    if(enforce_dc):
        output = masks * inputs + (1 - masks) * output

    output_crop = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(
            crop_320, x, dtype=tf.float32))(output)

    if(enforce_dc):
        model = keras.models.Model(
            inputs={
                'input': inputs, 'mask': masks}, outputs={
                'output': output, 'output_crop': output_crop})
    else:
        model = keras.models.Model(
            inputs=inputs,
            outputs={
                'output': output,
                'output_crop': output_crop})
    return model


def get_alternating_residual_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        enforce_dc):
    """Alternating model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns a frequency-space output of the same size, comprised of alternating frequency- and image-space convolutional layers and with connections from the input to each layer.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_convs(int): Number of convolutions per layer
      num_layers(int): Number of convolutional layers in model

    Returns:
      model: Keras model comprised of num_layers alternating image- and frequency-space convolutional layers with specified nonlinearities

    """
    inputs = Input(input_size)
    if(enforce_dc):
        masks = Input(input_size)

    n = inputs.get_shape().as_list()[1]
    inp_real = tf.expand_dims(inputs[:, :, :, 0], -1)
    inp_imag = tf.expand_dims(inputs[:, :, :, 1], -1)

    n_copies = int(num_features / 2)

    inp_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_real, inp_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    inputs_img = utils.convert_tensor_to_image_domain(inputs)
    inp_img_real = tf.expand_dims(inputs_img[:, :, :, 0], -1)
    inp_img_imag = tf.expand_dims(inputs_img[:, :, :, 1], -1)

    inp_img_copy = tf.reshape(tf.tile(tf.expand_dims(tf.concat(
        [inp_img_real, inp_img_imag], axis=3), 4), [1, 1, 1, 1, n_copies]), [-1, n, n, num_features])

    prev_layer = inputs

    for i in range(num_layers):
        for j in range(num_convs):
            k_conv = layers.BatchNormConv(
                num_features, kernel_size)(prev_layer) + inp_copy
            prev_layer = layers.get_nonlinear_layer(nonlinearity)(k_conv)

        prev_layer = utils.convert_channels_to_image_domain(prev_layer)

        for j in range(num_convs):
            img_conv = layers.BatchNormConv(
                num_features, kernel_size)(prev_layer) + inp_img_copy
            prev_layer = layers.get_nonlinear_layer('relu')(img_conv)

        prev_layer = utils.convert_channels_to_frequency_domain(prev_layer)

    output = Conv2D(2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer) + inputs

    if(enforce_dc):
        output = masks * inputs + (1 - masks) * output
        model = keras.models.Model(inputs=(inputs, masks), outputs=output)
    else:
        model = keras.models.Model(inputs=inputs, outputs=output)
    return model
