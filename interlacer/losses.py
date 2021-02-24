import sys

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K

from interlacer import utils
try:
    import lpips_tf
except:
    pass


def join_reim_mag_output(tensor):
    """

    Args:
      tensor: Tensor of shape (batch_size, n, n, 2)

    Returns:
      Tensor of shape (batch_size, n, n) with joined real and imag parts

    """
    return tf.expand_dims(K.abs(utils.join_reim_tensor(tensor)), -1)


def fourier_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is computed on Fourier space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'FREQ'):
        if(loss == 'L1'):
            def fourier_l1(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return K.mean(K.abs(y_true - y_pred))
            return fourier_l1
        elif(loss == 'L2'):
            def fourier_l2(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return fourier_l2
    elif(output_domain == 'IMAGE'):
        if(loss == 'L1'):
            def fourier_l1(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(
                    y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(
                    y_pred)
                y_true = utils.join_reim_tensor(y_true_fourier)
                y_pred = utils.join_reim_tensor(y_pred_fourier)
                return K.mean(K.abs(y_true - y_pred))
            return fourier_l1
        elif(loss == 'L2'):
            def fourier_l2(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(
                    y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(
                    y_pred)
                y_true = utils.join_reim_tensor(y_true_fourier)
                y_pred = utils.join_reim_tensor(y_pred_fourier)
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return fourier_l2


def comp_image_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is computed on real and imaginary components of image data.

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
                y_true = utils.convert_tensor_to_image_domain(y_true)
                y_pred = utils.convert_tensor_to_image_domain(y_pred)
                return K.mean(K.abs(y_true - y_pred))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                y_true = utils.convert_tensor_to_image_domain(y_true)
                y_pred = utils.convert_tensor_to_image_domain(y_pred)
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return image_l2


def image_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is computed on image space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'IMAGE'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return K.mean(K.abs(y_true - y_pred))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return image_l2
    elif(output_domain == 'FREQ'):
        if(loss == 'L1'):
            def image_l1(y_true, y_pred):
                y_true_image = utils.convert_tensor_to_image_domain(y_true)
                y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
                y_true = join_reim_mag_output(y_true_image)
                y_pred = join_reim_mag_output(y_pred_image)
                return K.mean(K.abs(y_true - y_pred))
            return image_l1
        elif(loss == 'L2'):
            def image_l2(y_true, y_pred):
                y_true_image = utils.convert_tensor_to_image_domain(y_true)
                y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
                y_true = join_reim_mag_output(y_true_image)
                y_pred = join_reim_mag_output(y_pred_image)
                return K.mean(K.pow(K.abs(y_true - y_pred), 2))
            return image_l2


def joint_img_freq_loss(output_domain, loss, loss_lambda):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is computed on both Fourier and image space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')
      loss_lambda(float): Weighting of freq loss vs image loss

    Returns:
      Function computing loss value from a true and predicted input

    """
    def joint_loss(y_true, y_pred):
        return(image_loss(output_domain, loss)(y_true, y_pred) + loss_lambda * fourier_loss(output_domain, loss)(y_true, y_pred))
    return joint_loss

if 'lpips_tf' in sys.modules:
    def lpips(output_domain):
        """Specifies a function which computes the appropriate loss function.

        Loss function here is SSIM on image-space data.

        Args:
          output_domain(str): Network output domain ('FREQ' or 'IMAGE')

        Returns:
          Function computing loss value from a true and predicted input

        """
        if(output_domain == 'IMAGE'):
            def image_lpips(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)

                y_true = K.tile(y_true, [1, 1, 1, 3])
                y_pred = K.tile(y_pred, [1, 1, 1, 3])

                return lpips_tf.lpips(y_true, y_pred, model='net-lin', net='alex')
            return image_lpips
        elif(output_domain == 'FREQ'):
            def image_lpips(y_true, y_pred):
                y_true_image = utils.convert_tensor_to_image_domain(y_true)
                y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
                y_true = join_reim_mag_output(y_true_image)
                y_pred = join_reim_mag_output(y_pred_image)

                y_true = K.tile(y_true, [1, 1, 1, 3])
                y_pred = K.tile(y_pred, [1, 1, 1, 3])
                return lpips_tf.lpips(
                    y_true, y_pred, model='net-lin', net='alex')
            return image_lpips


def joint_fastmri_loss(output_domain, loss):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is a combination of SSIM, PSNR, and componentwise error.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')
      loss(str): Loss type ('L1' or 'L2')

    Returns:
      Function computing loss value from a true and predicted input

    """
    def combined_loss(y_true, y_pred):
        return(ssim(output_domain)(y_true, y_pred) + 1 / 33.0 * psnr(output_domain)(y_true, y_pred) + 20 * comp_image_loss(output_domain, loss)(y_true, y_pred))
    return combined_loss


def ssim(output_domain):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is SSIM on image-space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'IMAGE'):
        def image_ssim(y_true, y_pred):
            y_true = join_reim_mag_output(y_true)
            y_pred = join_reim_mag_output(y_pred)
            return -1 * tf.image.ssim(y_true, y_pred,
                                      max_val=K.max(y_true), filter_size=7)
        return image_ssim
    elif(output_domain == 'FREQ'):
        def image_ssim(y_true, y_pred):
            y_true_image = utils.convert_tensor_to_image_domain(y_true)
            y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
            y_true = join_reim_mag_output(y_true_image)
            y_pred = join_reim_mag_output(y_pred_image)
            return -1 * tf.image.ssim(y_true, y_pred,
                                      max_val=K.max(y_true), filter_size=7)
        return image_ssim


def ssim_multiscale(output_domain):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is mulstiscale SSIM on image-space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'IMAGE'):
        def image_ssim_ms(y_true, y_pred):
            y_true = join_reim_mag_output(y_true)
            y_pred = join_reim_mag_output(y_pred)
            return -1 * \
                tf.image.ssim_multiscale(y_true, y_pred, max_val=K.max(y_true))
        return image_ssim_ms
    elif(output_domain == 'FREQ'):
        def image_ssim_ms(y_true, y_pred):
            y_true_image = utils.convert_tensor_to_image_domain(y_true)
            y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
            y_true = join_reim_mag_output(y_true_image)
            y_pred = join_reim_mag_output(y_pred_image)
            return -1 * \
                tf.image.ssim_multiscale(y_true, y_pred, max_val=K.max(y_true))
        return image_ssim_ms


def psnr(output_domain):
    """Specifies a function which computes the appropriate loss function.
    
    Loss function here is PSNR on image-space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'IMAGE'):
        def image_psnr(y_true, y_pred):
            y_true = join_reim_mag_output(y_true)
            y_pred = join_reim_mag_output(y_pred)
            return -1 * tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))
        return image_psnr
    elif(output_domain == 'FREQ'):
        def image_psnr(y_true, y_pred):
            y_true_image = utils.convert_tensor_to_image_domain(y_true)
            y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
            y_true = join_reim_mag_output(y_true_image)
            y_pred = join_reim_mag_output(y_pred_image)
            return -1 * tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))
        return image_psnr
