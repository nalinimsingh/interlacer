import numpy as np
import tensorflow as tf


def split_reim(array):
    """Split a complex valued matrix into its real and imaginary parts.

    Args:
      array(complex): An array of shape (batch_size, N, N) or (batch_size, N, N, 1)

    Returns:
      split_array(float): An array of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel

    """
    real = np.real(array)
    imag = np.imag(array)
    split_array = np.stack((real, imag), axis=3)
    return split_array


def split_reim_tensor(array):
    """Split a complex valued tensor into its real and imaginary parts.

    Args:
      array(complex): A tensor of shape (batch_size, N, N) or (batch_size, N, N, 1)

    Returns:
      split_array(float): A tensor of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel

    """
    real = tf.math.real(array)
    imag = tf.math.imag(array)
    split_array = tf.stack((real, imag), axis=3)
    return split_array


def split_reim_channels(array):
    """Split a complex valued tensor into its real and imaginary parts.

    Args:
      array(complex): A tensor of shape (batch_size, N, N) or (batch_size, N, N, 1)

    Returns:
      split_array(float): A tensor of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel

    """
    real = tf.math.real(array)
    imag = tf.math.imag(array)
    n_ch = array.get_shape().as_list()[3]
    split_array = tf.concat((real, imag), axis=3)
    return split_array


def join_reim(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.

    Args:
      array(float): An array of shape (batch_size, N, N, 2)

    Returns:
      joined_array(complex): An complex-valued array of shape (batch_size, N, N, 1)

    """
    joined_array = array[:, :, :, 0] + 1j * array[:, :, :, 1]
    return joined_array


def join_reim_tensor(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.

    Args:
      array(float): An array of shape (batch_size, N, N, 2)

    Returns:
      joined_array(complex): A complex-valued array of shape (batch_size, N, N)

    """
    joined_array = tf.cast(array[:, :, :, 0], 'complex64') + \
        1j * tf.cast(array[:, :, :, 1], 'complex64')
    return joined_array


def join_reim_channels(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.

    Args:
      array(float): An array of shape (batch_size, N, N, ch)

    Returns:
      joined_array(complex): A complex-valued array of shape (batch_size, N, N, ch/2)

    """
    ch = array.get_shape().as_list()[3]
    joined_array = tf.cast(array[:,
                                 :,
                                 :,
                                 :int(ch / 2)],
                           dtype=tf.complex64) + 1j * tf.cast(array[:,
                                                                    :,
                                                                    :,
                                                                    int(ch / 2):],
                                                              dtype=tf.complex64)
    return joined_array


def convert_to_frequency_domain(images):
    """Convert an array of images to their Fourier transforms.

    Args:
      images(float): An array of shape (batch_size, N, N, 2)

    Returns:
      spectra(float): An FFT-ed array of shape (batch_size, N, N, 2)

    """
    n = images.shape[1]
    spectra = split_reim(np.fft.fft2(join_reim(images), axes=(1, 2)))
    return spectra


def convert_tensor_to_frequency_domain(images):
    """Convert a tensor of images to their Fourier transforms.

    Args:
      images(float): A tensor of shape (batch_size, N, N, 2)

    Returns:
      spectra(float): An FFT-ed tensor of shape (batch_size, N, N, 2)

    """
    n = images.shape[1]
    spectra = split_reim_tensor(tf.signal.fft2d(join_reim_tensor(images)))
    return spectra


def convert_to_image_domain(spectra):
    """Convert an array of Fourier spectra to the corresponding images.

    Args:
      spectra(float): An array of shape (batch_size, N, N, 2)

    Returns:
      images(float): An IFFT-ed array of shape (batch_size, N, N, 2)

    """
    n = spectra.shape[1]
    images = split_reim(np.fft.ifft2(join_reim(spectra), axes=(1, 2)))
    return images


def convert_tensor_to_image_domain(spectra):
    """Convert an array of Fourier spectra to the corresponding images.

    Args:
      spectra(float): An array of shape (batch_size, N, N, 2)

    Returns:
      images(float): An IFFT-ed array of shape (batch_size, N, N, 2)

    """
    n = spectra.shape[1]
    images = split_reim_tensor(tf.signal.ifft2d(join_reim_tensor(spectra)))
    return images
