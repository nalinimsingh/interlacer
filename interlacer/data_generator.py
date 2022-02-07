import os

import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage.transform import resize
from tensorflow import keras

from scripts import filepaths
from interlacer import motion, utils


def normalize_slice(sl_data):
    """Normalize slice to z-scores across dataset.

    Args:
      sl_data(float): Input 2D numpy slice

    Returns:
      float: Normalized 2D output slice

    """
    sl_data = sl_data - sl_data.mean()
    norm_sl_data = sl_data / np.max(sl_data)
    return norm_sl_data


def get_mri_slices_from_dir(slice_dir):
    """Load and normalize MRI dataset.

    Args:
      slice_dir(str): Directory containing 2D MRI slices of shape (n, n); each slice is stored as a '.npz' file with keyword 'vol_data'

    Returns:
      float: A numpy array of size (num_images, n, n) containing all image slices

    """
    image_names = os.listdir(slice_dir)
    slices = []
    for img in image_names:
        vol_data = np.load(
            os.path.join(
                slice_dir,
                img),
            mmap_mode='r')['vol_data']
        sl_data = vol_data
        slices.append(sl_data)
    slices = np.asarray(slices)
    return slices


def get_mri_images():
    """Load normalized MRI training and validation images."""
    base_dir = filepaths.DATA_DIR
    train_slice_dir = os.path.join(base_dir, 'train/vols')
    val_slice_dir = os.path.join(base_dir, 'validate/vols')

    return get_mri_slices_from_dir(
        train_slice_dir), get_mri_slices_from_dir(val_slice_dir)


def get_mri_TEST_images():
    """Load normalized MRI test images."""
    base_dir = filepaths.DATA_DIR
    test_slice_dir = os.path.join(base_dir, 'test/vols')

    return get_mri_slices_from_dir(test_slice_dir)


def get_mri_spectra_stats(images):
    """Compute mean and stddev of MRI spectra.

    Args:
      images(float): Numpy array of shape (num_images, n, n) containing input images.

    Returns:
      float: Numpy array of shape (1, n, n, 2) containing pixelwise mean of the real and imaginary parts of the Fourier spectra of the input images
      float: Numpy array of shape (1, n, n, 2) containing pixelwise standard deviation of the real and imaginary parts of the Fourier spectra of the input images

    """
    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    spectra_mean = np.mean(spectra, axis=0, keepdims=True)
    spectra_std = np.clip(
        np.std(
            spectra,
            axis=0,
            keepdims=True),
        a_min=1,
        a_max=None)

    return spectra_mean, spectra_std


def generate_undersampled_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        enforce_dc,
        batch_size=10):
    """Generator that yields batches of undersampled input and correct output data.

    For corrupted inputs, select each line in k-space with probability corruption_frac and set it to zero.

    Args:
      images(float): Numpy array of input images, of shape (num_images, n, n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)

    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    num_batches = np.ceil(len(images) / batch_size)
    img_shape = images.shape[1]

    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    while True:
        n = images.shape[1]
        batch_inds = np.random.randint(0, images.shape[0], batch_size)

        if(input_domain=='MAG' or ('COMPLEX' in input_domain) ):
            n_ch_in = 1
        else:
            n_ch_in = 2

        if(output_domain=='MAG' or ('COMPLEX' in output_domain) ):
            n_ch_out = 1
        else:
            n_ch_out = 2
        inputs = np.empty((0, n, n, n_ch_in))
        outputs = np.empty((0, n, n, n_ch_out))
        masks = np.empty((0, n, n, n_ch_in))

        if('COMPLEX' in input_domain):
            masks = np.empty((0, n, n))

        for j in batch_inds:
            true_img = np.expand_dims(images[j, :, :, :], 0)
            true_k = np.expand_dims(spectra[j, :, :, :], 0)
            mask = np.ones(true_k.shape)

            img_size = images.shape[1]
            num_points = int(img_size * corruption_frac)
            coord_list = np.random.choice(
                range(img_size), num_points, replace=False)

            corrupt_k = true_k.copy()
            for k in range(len(coord_list)):
                corrupt_k[0, coord_list[k], :, :] = 0
                mask[0, coord_list[k], :, :] = 0
            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
                masks = np.append(masks, mask, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)
            elif(input_domain == 'MAG'):
                corrupt_img = np.expand_dims(np.abs(utils.join_reim(corrupt_img)),-1)
                inputs = np.append(inputs, corrupt_img / nf, axis=0)
            elif(input_domain == 'COMPLEX_K'):
                corrupt_k = np.expand_dims(utils.join_reim(corrupt_k),-1)
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
            elif(input_domain == 'COMPLEX_I'):
                corrupt_img = np.expand_dims(utils.join_reim(corrupt_img),-1)
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)
            elif(output_domain == 'MAG'):
                true_img = np.expand_dims(np.abs(utils.join_reim(true_img)),-1)
                outputs = np.append(outputs, true_img / nf, axis=0)
            elif(output_domain == 'COMPLEX_K'):
                true_k = np.expand_dims(utils.join_reim(true_k),-1)
                outputs = np.append(inputs, true_k / nf, axis=0)
            elif(output_domain == 'COMPLEX_I'):
                true_img = np.expand_dims(utils.join_reim(true_img),-1)
                outputs = np.append(inputs, true_img / nf, axis=0)

            if('COMPLEX' in input_domain):
                mask = mask [:,:,:,0]
                masks = np.append(masks, mask, axis=0)

        if(enforce_dc):
            yield((inputs, masks), outputs)
        else:
            yield(inputs, outputs)


def generate_uniform_undersampled_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        enforce_dc,
        batch_size=10):
    """Generator that yields batches of undersampled input and correct output data.

    For corrupted inputs, select each line in k-space with probability corruption_frac and set it to zero.

    Args:
      images(float): Numpy array of input images, of shape (num_images, n, n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)

    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    num_batches = np.ceil(len(images) / batch_size)
    img_shape = images.shape[1]

    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    while True:
        n = images.shape[1]
        batch_inds = np.random.randint(0, images.shape[0], batch_size)

        if(input_domain=='MAG' or ('COMPLEX' in input_domain) ):
            n_ch_in = 1
        else:
            n_ch_in = 2

        if(output_domain=='MAG' or ('COMPLEX' in output_domain) ):
            n_ch_out = 1
        else:
            n_ch_out = 2
        inputs = np.empty((0, n, n, n_ch_in))
        outputs = np.empty((0, n, n, n_ch_out))
        masks = np.empty((0, n, n, n_ch_in))

        if('COMPLEX' in input_domain):
            masks = np.empty((0, n, n))

        for j in batch_inds:
            true_img = np.expand_dims(images[j, :, :, :], 0)
            true_k = np.expand_dims(spectra[j, :, :, :], 0)
            mask = np.ones(true_k.shape)

            img_size = images.shape[1]
            num_points = int(img_size * corruption_frac)

            s = int(1/(1-corruption_frac))
            arc_lines = int(32/s)

            arc_low = int((50-int(arc_lines/2))*n/100)
            arc_high = int((50+int(arc_lines/2))*n/100)

            coord_list_low = np.concatenate([[i+j for j in range(s-1)] for i in range(0,arc_low,s)],axis=0)
            coord_list_high = np.concatenate([[i+j for j in range(s-1)] for i in range(arc_high,n-(s-1),s)],axis=0)
            coord_list = np.concatenate([coord_list_low,coord_list_high],axis=0)

            corrupt_k = true_k.copy()
            for k in range(len(coord_list)):
                corrupt_k[0, coord_list[k], :, :] = 0
                mask[0, coord_list[k], :, :] = 0
            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
                masks = np.append(masks, mask, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)
            elif(input_domain == 'MAG'):
                corrupt_img = np.expand_dims(np.abs(utils.join_reim(corrupt_img)),-1)
                inputs = np.append(inputs, corrupt_img / nf, axis=0)
            elif(input_domain == 'COMPLEX_K'):
                corrupt_k = np.expand_dims(utils.join_reim(corrupt_k),-1)
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
            elif(input_domain == 'COMPLEX_I'):
                corrupt_img = np.expand_dims(utils.join_reim(corrupt_img),-1)
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)
            elif(output_domain == 'MAG'):
                true_img = np.expand_dims(np.abs(utils.join_reim(true_img)),-1)
                outputs = np.append(outputs, true_img / nf, axis=0)
            elif(output_domain == 'COMPLEX_K'):
                true_k = np.expand_dims(utils.join_reim(true_k),-1)
                outputs = np.append(inputs, true_k / nf, axis=0)
            elif(output_domain == 'COMPLEX_I'):
                true_img = np.expand_dims(utils.join_reim(true_img),-1)
                outputs = np.append(inputs, true_img / nf, axis=0)

            if('COMPLEX' in input_domain):
                mask = mask [:,:,:,0]
                masks = np.append(masks, mask, axis=0)

        if(enforce_dc):
            yield((inputs, masks), outputs)
        else:
            yield(inputs, outputs)


def generate_motion_data(
        images,
        input_domain,
        output_domain,
        mot_frac,
        max_htrans,
        max_vtrans,
        max_rot,
        batch_size=10):
    """Generator that yields batches of motion-corrupted input and correct output data.

    For corrupted inputs, select some lines at which motion occurs; randomly generate and apply translation/rotations at those lines.

    Args:
      images(float): Numpy array of input images, of shape (num_images,n,n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      mot_frac(float): Fraction of lines at which motion occurs.
      max_htrans(float): Maximum fraction of image width for a translation.
      max_vtrans(float): Maximum fraction of image height for a translation.
      max_rot(float): Maximum fraction of 360 degrees for a rotation.
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)

    Returns:
      inputs: Tuple of corrupted input data and correct output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    num_batches = np.ceil(len(images) / batch_size)
    img_shape = images.shape[1]

    reim_images = images.copy()
    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    while True:
        n = images.shape[1]
        batch_inds = np.random.randint(0, images.shape[0], batch_size)

        inputs = np.empty((0, n, n, 2))
        outputs = np.empty((0, n, n, 2))
        masks = np.empty((0, n, n, 2))

        for j in batch_inds:
            true_img = np.expand_dims(images[j, :, :, :], 0)

            img_size = images.shape[1]
            num_points = int(mot_frac * n)
            coord_list = np.sort(
                np.random.choice(
                    img_size,
                    size=num_points,
                    replace=False))
            num_pix = np.zeros((num_points, 2))
            angle = np.zeros(num_points)

            max_htrans_pix = n * max_htrans
            max_vtrans_pix = n * max_vtrans
            max_rot_deg = 360 * max_rot

            num_pix[:, 0] = np.random.random(
                num_points) * (2 * max_htrans_pix) - max_htrans_pix
            num_pix[:, 1] = np.random.random(
                num_points) * (2 * max_vtrans_pix) - max_vtrans_pix
            angle = np.random.random(num_points) * \
                (2 * max_rot_deg) - max_rot_deg

            corrupt_k, true_k = motion.add_rotation_and_translations(
                reim_images[j, :, :], coord_list, angle, num_pix)
            corrupt_k = utils.split_reim(np.expand_dims(corrupt_k, 0))
            true_k = utils.split_reim(np.expand_dims(true_k, 0))

            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        yield(inputs, outputs)


def generate_noisy_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        batch_size=10):
    """Generator that yields batches of noisy input and correct output data.

    For corrupted inputs, add complex-valued noise with standard deviation corruption_frac at each pixel in k-space.

    Args:
      images(float): Numpy array of input images, of shape (num_images,n,n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Variance of complex-valued noise to be added
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)

    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    num_batches = np.ceil(len(images) / batch_size)
    img_shape = images.shape[1]

    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    while True:
        n = images.shape[1]
        batch_inds = np.random.randint(0, images.shape[0], batch_size)

        inputs = np.empty((0, n, n, 2))
        outputs = np.empty((0, n, n, 2))
        masks = np.empty((0, n, n, 2))

        for j in batch_inds:
            true_img = np.expand_dims(images[j, :, :, :], 0)
            true_k = np.expand_dims(spectra[j, :, :, :], 0)
            mask = np.ones(true_k.shape)

            img_size = images.shape[1]
            noise = np.random.normal(
                loc=0.0, scale=corruption_frac, size=true_k.shape)

            corrupt_k = true_k.copy() + noise
            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
                masks = np.append(masks, mask, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        yield(inputs, outputs)


def generate_undersampled_motion_data(
        images,
        input_domain,
        output_domain,
        us_frac,
        mot_frac,
        max_htrans,
        max_vtrans,
        max_rot,
        batch_size=10):
    """Generator that yields batches of motion-corrupted, undersampled input and correct output data.

    For corrupted inputs, select some lines at which motion occurs; randomly generate and apply translation/rotations at those lines.

    Args:
      images(float): Numpy array of input images, of shape (num_images,n,n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      us_frac(float): Fraction of lines at which motion occurs.
      mot_frac(float): Fraction of lines at which motion occurs.
      max_htrans(float): Maximum fraction of image width for a translation.
      max_vtrans(float): Maximum fraction of image height for a translation.
      max_rot(float): Maximum fraction of 360 degrees for a rotation.
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)

    Returns:
      inputs: Tuple of corrupted input data and correct output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    def get_us_motion_mask(arr_shape, us_frac):
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

            return mask.T

        else:
            return(np.ones(arr_shape)).T

    num_batches = np.ceil(len(images) / batch_size)
    img_shape = images.shape[1]

    reim_images = images.copy()
    images = utils.split_reim(images)
    spectra = utils.convert_to_frequency_domain(images)

    while True:
        n = images.shape[1]
        batch_inds = np.random.randint(0, images.shape[0], batch_size)

        inputs = np.empty((0, n, n, 2))
        outputs = np.empty((0, n, n, 2))
        masks = np.empty((0, n, n, 2))

        for j in batch_inds:

            true_img = np.expand_dims(images[j, :, :, :], 0)

            img_size = images.shape[1]
            num_points = int(np.random.random() * mot_frac * n)
            coord_list = np.sort(
                np.random.choice(
                    img_size,
                    size=num_points,
                    replace=False))
            num_pix = np.zeros((num_points, 2))
            angle = np.zeros(num_points)

            max_htrans_pix = n * max_htrans
            max_vtrans_pix = n * max_vtrans
            max_rot_deg = 360 * max_rot

            num_pix[:, 0] = np.random.random(
                num_points) * (2 * max_htrans_pix) - max_htrans_pix
            num_pix[:, 1] = np.random.random(
                num_points) * (2 * max_vtrans_pix) - max_vtrans_pix
            angle = np.random.random(num_points) * \
                (2 * max_rot_deg) - max_rot_deg

            corrupt_k, true_k = motion.add_rotation_and_translations(
                reim_images[j, :, :], coord_list, angle, num_pix)
            true_k = utils.split_reim(np.expand_dims(true_k, 0))
            true_img = utils.convert_to_image_domain(true_k)
            corrupt_k = utils.split_reim(np.expand_dims(corrupt_k, 0))

            mask = get_us_motion_mask(true_img.shape[1:3], us_frac)
            r_mask = np.expand_dims(
                np.repeat(mask[:, :, np.newaxis], 2, axis=-1), 0)

            corrupt_k *= r_mask
            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        yield(inputs, outputs)


def generate_data(
        images,
        exp_config,
        split=None):
    """Return a generator with corrupted and corrected data.

    Args:
      images: float
      task(str): 'undersample' (no other tasks supported for FastMRI data)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      batch_size(int, optional): Number of input-output pairs in each batch
      split(str): Which data split to use ('train', 'val', 'test')

    Returns:
      generator yielding a tuple containing a single batch of corrupted and corrected data

    """
    task = exp_config.task
    input_domain = exp_config.input_domain
    output_domain = exp_config.output_domain
    batch_size = exp_config.batch_size

    if(task == 'undersample'):
        return generate_undersampled_data(
            images,
            input_domain,
            output_domain,
            exp_config.us_frac,
            exp_config.enforce_dc,
            batch_size)
    elif(task == 'uniform_undersample'):
        return generate_uniform_undersampled_data(
            images,
            input_domain,
            output_domain,
            exp_config.us_frac,
            exp_config.enforce_dc,
            batch_size)
    elif(task == 'motion'):
        return generate_motion_data(
            images,
            input_domain,
            output_domain,
            exp_config.mot_frac,
            exp_config.max_htrans,
            exp_config.max_vtrans,
            exp_config.max_rot,
            batch_size)
    elif(task == 'noise'):
        return generate_noisy_data(
            images,
            input_domain,
            output_domain,
            exp_config.noise_std,
            batch_size)
    elif(task == 'undersample_motion'):
        return generate_undersampled_motion_data(
            images,
            input_domain,
            output_domain,
            exp_config.us_frac,
            exp_config.mot_frac,
            exp_config.max_htrans,
            exp_config.max_vtrans,
            exp_config.max_rot,
            batch_size)
