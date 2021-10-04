import os
import time

import h5py
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from interlacer import models, utils
from scripts import filepaths

n_crop = 320

def get_fastmri_slices_from_dir(
        image_dir,
        batch_size,
        fs=False,
        corruption_frac=1.0):
    """Load and normalize MRI dataset.

    Args:
      image_dir(str): Directory containing 3D MRI volumes of shape (?, n, n); each volume is stored as a '.h5' file in the FastMRI format
      batch_size(int): Number of input-output pairs in each batch
      fs(Boolean): Whether to read images with fat suppression (True) or without (False)
      corruption_frac(float): Probability with which to zero a line in k-space

    Returns:
      float: A numpy array of size (num_images, n, n) containing all image slices

    """
    image_names = os.listdir(image_dir)

    found_vol = False
    while not found_vol:
        img_i = np.random.randint(0, len(image_names))
        img = image_names[img_i]

        with h5py.File(os.path.join(image_dir, img), "r") as f:
            if (('CORPDFS' in f.attrs['acquisition']) == fs):
                n_slices = f['kspace'].shape[0]

                kspace_fulls = np.empty((batch_size,
                                         f['kspace'].shape[1],
                                         f['kspace'].shape[2]),
                                         dtype=complex)
                masks = np.empty((batch_size,
                                  f['kspace'].shape[1],
                                  f['kspace'].shape[2]))
                kspace_masks = kspace_fulls.copy()
                kspace_full_crops = np.empty((batch_size,n_crop,
                                              n_crop),dtype=complex)

                for i in range(batch_size):
                    slice_i = np.random.randint(0, n_slices)

                    kspace_full = f['kspace'][slice_i, :, :]

                    mask = get_undersampling_mask(
                        kspace_full.shape, corruption_frac)

                    kspace_mask = mask*f['kspace'][slice_i, :, :]
                    kspace_full_crop = models.crop_320(
                            utils.split_reim(np.expand_dims(kspace_full,0))[0,:,:,:])
                    kspace_full_crop = utils.join_reim(np.expand_dims(kspace_full_crop,0))[0,:,:]

                    kspace_fulls[i,...] = kspace_full
                    kspace_full_crops[i,...] = kspace_full_crop
                    kspace_masks[i,...] = kspace_mask
                    masks[i,...] = mask

                found_vol = True

    return kspace_fulls, kspace_full_crops, kspace_masks, masks


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
        batch_size,
        fs=False):
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
        kspace_full, kspace_full_crop, kspace_mask, mask = get_fastmri_slices_from_dir(
            image_dir, batch_size, fs, corruption_frac=corruption_frac)

        mask = np.expand_dims(mask,-1)
        mask = np.repeat(mask,2,axis=-1)

        kspace_full = utils.split_reim(kspace_full)
        kspace_full_crop = utils.split_reim(kspace_full_crop)
        kspace_mask = utils.split_reim(kspace_mask)

        # Bring majority of values to 0-1 range.
        corrupt_img = utils.convert_to_image_domain(kspace_mask)
        nf = np.percentile(np.abs(corrupt_img), 95, axis=(1,2,3))
        nf = nf[:,np.newaxis, np.newaxis, np.newaxis]

        if(input_domain == 'FREQ'):
            inp = kspace_mask / nf
        elif(input_domain == 'IMAGE'):
            inp = utils.convert_to_image_domain(kspace_mask) / nf

        if(output_domain == 'FREQ'):
            output = kspace_full / nf
            output_crop = kspace_full_crop / nf
        elif(output_domain == 'IMAGE'):
            output = utils.convert_to_image_domain(kspace_mask) / nf
            output_crop = utils.convert_to_image_domain(kspace_full_crop) / nf

        if(enforce_dc):
            yield({'input':inp, 'mask':mask},
                  {'output':output, 'output_crop':output_crop})
        else:
            yield(inp, {'output':output, 'output_crop':output_crop})


def generate_data(
        image_dir,
        exp_config,
        batch_size=4,
        fs=False):
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
            batch_size,
            fs=fs)
