import os
import time

import h5py
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from interlacer import motion, utils
from scripts import filepaths


def get_fastmri_slices_from_dir(
        image_dir,
        batch_size,
        fs=False,
        corruption_frac=1.0,
        return_masks=False):
    """Load and normalize MRI dataset.

    Args:
      image_dir(str): Directory containing 3D MRI volumes of shape (?, n, n); each volume is stored as a '.h5' file in the FastMRI format
      batch_size(int): Number of input-output pairs in each batch
      fs(Boolean): Whether to read images with fat suppression (True) or without (False)
      corruption_frac(float): Probability with which to zero a line in k-space
      return_masks(Boolean): Whether to return k-space masks for all images in batch

    Returns:
      float: A numpy array of size (num_images, n, n) containing all image slices

    """
    image_names = os.listdir(image_dir)
    slices = []
    kspace = []
    masks = []

    batch_inds = np.random.randint(0, len(image_names), batch_size)

    i = 0
    while(i < batch_size):
        img_i = np.random.randint(0, len(image_names))
        img = image_names[img_i]

        with h5py.File(os.path.join(image_dir, img), "r") as f:
            if (('CORPDFS' in f.attrs['acquisition']) == fs):
                n_slices = f['kspace'].shape[0]

                slice_i = np.random.randint(0, n_slices)

                sl_img = f['reconstruction_rss'][slice_i, :, :]
                n = int(sl_img.shape[0] / 2)

                sl_k = f['kspace'][slice_i, :, :]

                mask = get_undersampling_mask(
                    sl_k.shape, corruption_frac)

                # Bring k-space down to square size
                sl_k = np.fft.ifftshift(sl_k)
                sl_img_unshift = np.fft.fftshift(np.fft.ifft2(sl_k))
                x_mid = int(sl_img_unshift.shape[0] / 2)
                y_mid = int(sl_img_unshift.shape[1] / 2)
                sl_img_crop = sl_img_unshift[x_mid -
                                             n:x_mid + n, y_mid - n:y_mid + n]
                mask_crop = mask[x_mid - n:x_mid + n, y_mid - n:y_mid + n]
                mask_crop = np.fft.fftshift(mask_crop)
                sl_k = np.fft.fft2(sl_img_crop)

                slices.append(sl_img)
                kspace.append(sl_k)
                masks.append(mask_crop)
                i += 1
            else:
                pass
    slices = np.asarray(slices)
    kspace = np.asarray(kspace)
    masks = np.asarray(masks)
    if(return_masks):
        return slices, kspace, masks
    else:
        return slices, kspace


def get_undersampling_mask(arr_shape, us_frac):
    """ Based on https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py. """

    num_cols = arr_shape[1]
    if(us_frac != 1):
        acceleration = int(1 / (1 - us_frac))
        center_fraction = (1 - us_frac) * 0.08 / 0.25

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / \
            (num_cols - num_low_freqs)
        mask_inds = np.random.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask_inds[pad:pad + num_low_freqs] = True

        mask = np.zeros(arr_shape)
        mask[:, mask_inds] = 1

        return mask

    else:
        return(np.ones(arr_shape))


def generate_undersampled_data(
        image_dir,
        input_domain,
        output_domain,
        corruption_frac,
        enforce_dc,
        fs=False,
        batch_size=16):
    """Generator that yields batches of undersampled input and correct output data.

    For corrupted inputs, select each line in k-space with probability corruption_frac and set it to zero.

    Args:
      image_dir(str): Directory containing 3D MRI volumes
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      fs(Bool, optional): Whether to read images with fat suppression (True) or without (False)
      batch_size(int, optional): Number of input-output pairs in each batch

    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).

    """

    while True:
        images, kspace, masks = get_fastmri_slices_from_dir(
            image_dir, batch_size, fs, corruption_frac=corruption_frac, return_masks=True)

        images = utils.split_reim(images)
        spectra = utils.convert_to_frequency_domain(images)

        n = images.shape[1]

        inputs = np.empty((0, n, n, 2))
        outputs = np.empty((0, n, n, 2))
        r_masks = np.empty((0, n, n, 2))

        for j in range(batch_size):
            true_img = np.expand_dims(images[j, :, :, :], 0)
            true_k = np.expand_dims(spectra[j, :, :, :], 0)
            mask = masks[j, :, :]
            r_mask = np.expand_dims(
                np.repeat(mask[:, :, np.newaxis], 2, axis=-1), 0)

            num_points = int(n * corruption_frac)
            coord_list = np.random.choice(
                n, num_points, replace=False)

            corrupt_k = kspace[j, :, :] * mask

            # Bring majority of values to 0-1 range.
            corrupt_k = utils.split_reim(np.expand_dims(corrupt_k, 0)) * 500

            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(np.abs(corrupt_img))

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
                r_masks = np.append(r_masks, r_mask, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        if(enforce_dc):
            yield((inputs, r_masks), outputs)
        else:
            yield(inputs, outputs)


def generate_data(
        image_dir,
        exp_config,
        fs=False,
        batch_size=16):
    """Return a generator with corrupted and corrected data.

    Args:
      image_dir(str): Directory containing 3D MRI volumes
      task(str): 'undersample' (no other tasks supported for FastMRI data)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      fs(Bool, optional): Whether to read images with fat suppression (True) or without (False)
      batch_size(int, optional): Number of input-output pairs in each batch

    Returns:
      generator yielding a tuple containing a single batch of corrupted and corrected data

    """
    task = exp_config.task
    input_domain = exp_config.input_domain
    output_domain = exp_config.output_domain
    us_frac = exp_config.us_frac
    enforce_dc = exp_config.enforce_dc
    batch_size = exp_config.batch_size

    if(task == 'undersample'):
        return generate_undersampled_data(
            image_dir,
            input_domain,
            output_domain,
            us_frac,
            enforce_dc,
            fs=fs,
            batch_size=batch_size)
