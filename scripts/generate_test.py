"""Script to automatically write h5 files containing test set data

  Assumes that test volumes are stored in test/vols under filepaths.DATA_DIR

  Usage:

  $ python generate_test.py
 
"""

import h5py
import numpy as np
import os

import filepaths
from interlacer import data_generator, fastmri_data_generator


batch_size = 100
us_frac = 0.75
mot_frac = 0.03
max_htrans = 0.03
max_vtrans = 0.03
max_rot = 0.03
noise_std = 10000

input_domain = 'FREQ'
output_domain = 'FREQ'

base_dir = filepaths.DATA_DIR
test_slice_dir = os.path.join(base_dir, 'test/vols')
images = data_generator.get_mri_slices_from_dir(test_slice_dir)

data_generators = [data_generator.generate_undersampled_data(
            images,
            input_domain,
            output_domain,
            us_frac,
            True,
            batch_size),

            data_generator.generate_motion_data(
            images,
            input_domain,
            output_domain,
            mot_frac,
            max_htrans,
            max_vtrans,
            max_rot,
            batch_size),

            data_generator.generate_noisy_data(
            images,
            input_domain,
            output_domain,
            noise_std,
            batch_size),

            data_generator.generate_undersampled_motion_data(
            images,
            input_domain,
            output_domain,
            us_frac,
            mot_frac,
            max_htrans,
            max_vtrans,
            max_rot,
            batch_size),

            data_generator.generate_uniform_undersampled_data(
            images,
            input_domain,
            output_domain,
            us_frac,
            True,
            batch_size)]

filenames = ['brain_undersample_test.h5','brain_motion_test.h5','brain_noise_test.h5','brain_undersample_motion_test.h5','brain_uniform_undersample_8x_test.h5']

if not(os.path.exists('testsets')):
    os.mkdir('testsets')

for i in [4]:#range(4):
    dg = data_generators[i]
    fn = os.path.join('testsets/',filenames[i])

    if('undersample_test' in fn or 'undersample_8x_test' in fn):
        (m_in, m_mask), m_out = next(dg)
    else:
        m_in, m_out = next(dg)

    file = h5py.File(fn, "w")

    dataset = file.create_dataset(
        "inputs", np.shape(m_in), data=m_in
    )
    if('undersample_test' in fn or 'undersample_8x_test' in fn):
        dataset = file.create_dataset(
            "masks", np.shape(m_mask), data=m_mask
        )
    dataset = file.create_dataset(
        "outputs", np.shape(m_out), data=m_out
    )
    file.close()
