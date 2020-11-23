import os

import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from scripts import filepaths
from interlacer import motion, utils


def get_mnist_images():
    """Load MNIST images, normalized to [0,1]."""
    (img_train, _), (img_test, _) = mnist.load_data()
    img_train = img_train / 255.
    img_test = img_test / 255.
    return img_train, img_test


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

        inputs = np.empty((0, n, n, 2))
        outputs = np.empty((0, n, n, 2))
        masks = np.empty((0, n, n, 2))

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

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        yield(inputs, outputs)


def generate_motion_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        batch_size=10):
    """Generator that yields batches of motion-corrupted input and correct output data.

    For corrupted inputs, select some lines at which motion occurs; randomly generate and apply translation/rotations at those lines.

    Args:
      images(float): Numpy array of input images, of shape (num_images,n,n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Fraction of lines at which motion occurs.
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
            num_points = int(corruption_frac * n)
            coord_list = np.sort(
                np.random.choice(
                    img_size,
                    size=num_points,
                    replace=False))
            num_pix = np.zeros((num_points, 2))
            angle = np.zeros(num_points)

            num_pix[:, 0] = np.random.random(num_points) * 40 - 20
            num_pix[:, 1] = np.random.random(num_points) * 40 - 20
            angle = np.random.random(num_points) * 30 - 15

            corrupt_k, true_k = motion.add_rotation_and_translations(
                reim_images[j, :, :], coord_list, angle, num_pix)
            corrupt_k = utils.split_reim(np.expand_dims(corrupt_k, 0))
            true_k = utils.split_reim(np.expand_dims(true_k, 0))

            corrupt_img = utils.convert_to_image_domain(corrupt_k)

            nf = np.max(corrupt_img)

            if(input_domain == 'FREQ'):
                inputs = np.append(inputs, corrupt_k / nf, axis=0)
                #masks = np.append(masks, mask, axis=0)
            elif(input_domain == 'IMAGE'):
                inputs = np.append(inputs, corrupt_img / nf, axis=0)

            if(output_domain == 'FREQ'):
                outputs = np.append(outputs, true_k / nf, axis=0)
            elif(output_domain == 'IMAGE'):
                outputs = np.append(outputs, true_img / nf, axis=0)

        yield(inputs, outputs)


def generate_stored_motion_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        batch_size=10,
        split='None'):
    """Generator that reads stored batches of motion-corrupted input and correct output data.

    Args:
      images(float): Numpy array of input images, of shape (num_images,n,n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Fraction of lines at which motion occurs.
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)
      split:  (Default value = 'None')

    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).

    """
    num_batches = np.int(images.shape[0] / batch_size)
    base_dir = filepaths.MOTION_DATA_DIR
    data_dir = os.path.join(base_dir, split)

    while True:
        i = np.random.randint(0, num_batches)
        dir_str = os.path.join(
            data_dir,
            input_domain +
            '-' +
            input_domain +
            '-' +
            str(corruption_frac) +
            '-' +
            str(batch_size))

        data = np.load(os.path.join(dir_str, str(i) + '.npz'))
        m_input = data['m_input']
        m_label = data['m_label']

        yield(m_input, m_label)


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


def generate_data(
        images,
        task,
        input_domain,
        output_domain,
        corruption_frac,
        batch_size=16,
        split=None):
    """Return a generator with corrupted and corrected data.

    Args:
      images: float
      task: str
      input_domain: str
      output_domain: str
      corruption_frac: float
      batch_size: int (Default value = 16)
      split: str (Default value = None)

    Returns:
      generator yielding a tuple containing a single batch of corrupted and corrected data

    """
    if(task == 'undersample'):
        return generate_undersampled_data(
            images,
            input_domain,
            output_domain,
            corruption_frac,
            batch_size)
    elif(task == 'motion'):
        return generate_stored_motion_data(
            images,
            input_domain,
            output_domain,
            corruption_frac,
            batch_size,
            split=split)
    elif(task == 'noise'):
        return generate_noisy_data(
            images,
            input_domain,
            output_domain,
            corruption_frac,
            batch_size)
