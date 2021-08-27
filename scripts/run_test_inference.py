"""Script to automatically run inference on a specified test set

Writes the outputs as another h5 to the trained model directory.

  Usage:

  $ python run_test_inference.py undersample_exp testsets/undersample_test.h5

"""

import argparse
import h5py
import os
import numpy as np

import filepaths
from interlacer import utils
from visualizations import visualization_lib

import tensorflow as tf
from tensorflow import image

parser = argparse.ArgumentParser(
    description='Run inference on test set.')
parser.add_argument(
    'exp_str',
    help='Number of experiment directory')
parser.add_argument(
    'test_file',
    help='Location of h5 containing the test set')

args = parser.parse_args()


exp_str = 'undersample_motion'
test_data_path = args.test_file
exp_dir = filepaths.TRAIN_DIR + args.exp_str

f = h5py.File(test_data_path, 'r')
freq_inputs = f['inputs'][()]
if(exp_str=='undersample'):
    masks = f['masks'][()]
img_inputs = utils.convert_to_image_domain(freq_inputs)
mag_inputs = np.expand_dims(np.abs(utils.join_reim(img_inputs)),-1)

freq_label = f['outputs'][()]
img_label = utils.convert_to_image_domain(freq_label)
mag_label = np.expand_dims(np.abs(utils.join_reim(img_label)),-1)


for exp in os.listdir(exp_dir):
    model_dir = os.path.join(exp_dir, exp)
    config_path = os.path.join(model_dir,visualization_lib.get_config_path(model_dir))
    config, model = visualization_lib.load_model(config_path)

    best_ckpt = visualization_lib.get_best_ckpt(model_dir)
    model.load_weights(os.path.join(model_dir,'cp-'+str(best_ckpt).zfill(4)+'.ckpt'))

    if('FREQ' in exp):
        if(exp_str=='undersample'):
            freq_outputs = model.predict((freq_inputs,masks))
        else:
            freq_outputs = model.predict(freq_inputs)
        img_outputs = utils.convert_to_image_domain(freq_outputs)
    else:
        img_outputs = model.predict(img_inputs)
        freq_outputs = utils.convert_to_frequency_domain(img_outputs)

    mag_outputs = np.expand_dims(np.abs(utils.join_reim(img_outputs)),-1)

    ssim = image.ssim(
        tf.convert_to_tensor(mag_outputs, dtype=tf.float32), 
        tf.convert_to_tensor(mag_label, dtype=tf.float32), 
        np.max(mag_label), filter_size=11, filter_sigma=1.5, 
        k1=0.01, k2=0.03).numpy()
    psnr = image.psnr(
        tf.convert_to_tensor(mag_outputs, dtype=tf.float32), 
        tf.convert_to_tensor(mag_label, dtype=tf.float32), 
        np.max(mag_label)).numpy()

    output_file = os.path.join(model_dir,exp_str+'_test_output.h5')
    if os.path.exists(output_file):
        os.remove(output_file)
    file = h5py.File(output_file, "w")

    dataset = file.create_dataset(
        "img_outputs", np.shape(img_outputs), data=img_outputs
    )
    dataset = file.create_dataset(
        "freq_outputs", np.shape(freq_outputs), data=freq_outputs
    )
    dataset = file.create_dataset(
        "mag_outputs", np.shape(mag_outputs), data=mag_outputs
    )
    dataset = file.create_dataset(
        "ssim", np.shape(ssim), data=ssim
    )
    dataset = file.create_dataset(
        "psnr", np.shape(psnr), data=psnr
    )
    file.close()

