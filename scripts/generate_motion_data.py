"""Script to precompute data for motion correction task.

  For a given batch size, computes motion-corrupted input-output pairs in both the image- and frequency- space based on interlacer.data_generator.generate_motion_data() and writes these precomputed batches to filepaths.MOTION_DATA_DIR.

  Usage:

    $ python generate_motion_data.py 16

  Options:

    batch_size(int): Size of batches to be written.
"""

import argparse
import os
import sys

import numpy as np

import interlacer.data_generator as data_generator
import scripts.filepaths as filepaths

parser = argparse.ArgumentParser(
    description='Precompute and store data for motion correction experiments.')
parser.add_argument(
    'batch_size',
    help='Number of images per batch, all read in together.')

args = parser.parse_args()
batch_size = int(args.batch_size)

datasets = ['train', 'val', 'test']
domains = ['FREQ', 'IMAGE']
corruption_fracs = [0.01, 0.03, 0.05]

base_dir = filepaths.MOTION_DATA_DIR

for dataset in datasets:
    if(dataset == 'train'):
        images, _ = data_generator.get_mri_images()
    elif(dataset == 'val'):
        _, images = data_generator.get_mri_images()
    elif(dataset == 'test'):
        images = data_generator.get_mri_TEST_images()
        

    data_dir = os.path.join(base_dir, dataset)
    for domain in domains:
        for corruption_frac in corruption_fracs:
            dir_str = os.path.join(
                data_dir,
                domain +
                '-' +
                domain +
                '-' +
                str(corruption_frac) +
                '-' +
                str(batch_size))
            if not os.path.exists(dir_str):
                os.makedirs(dir_str)
            generator = data_generator.generate_motion_data(
                images, domain, domain, corruption_frac, batch_size=batch_size)

            for i in range(int(images.shape[0] / batch_size)):
                m_input, m_label = next(generator)

                outfile = os.path.join(dir_str, str(i))
                np.savez(outfile, m_input=m_input, m_label=m_label)
