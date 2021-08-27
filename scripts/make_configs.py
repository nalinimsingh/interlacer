"""Script to automatically generate config files for training.

After editing the lists under 'Customizable fields' with desired experiment configurations, running this script will create a directory called $exp_name at the path in scripts/filepaths.CONFIG_DIR. This script then populates the directory with all combinations of the specified fields, except for those under 'Excluded configs'.

  Usage:

  $ python make_configs.py
  
"""

import itertools
import math
import os
import shutil

import numpy as np

import filepaths

exp_name = 'cross_model_comparison'

# Customizable fields
datasets = ['MRI']
tasks = ['undersample_motion']
us_fracs = ['0.75']
mot_fracs = ['0.03']
max_htranses = ['0.03']
max_vtranses = ['0.03']
max_rots = ['0.03']
noise_stds = ['None']

architectures = [
    'CONV_RESIDUAL',
    'INTERLACER_RESIDUAL',
    'ALTERNATING_RESIDUAL']
kernel_sizes = ['9']
num_featureses = ['32']
num_convses = ['1']
num_layerses = ['10','20']
loss_types = ['compimage']
losses = ['L1']
loss_lambdas = ['0.1']
input_domains = ['IMAGE', 'FREQ']
output_domains = ['IMAGE', 'FREQ']
nonlinearities = ['relu', '3-piece']
enforce_dcs = ['True', 'False']

num_epochses = ['5000']
batch_sizes = ['4']


for dataset, task, us_frac, mot_frac, max_htrans, max_vtrans, max_rot, noise_std, architecture, kernel_size, num_features, num_convs, num_layers, loss_type, loss, loss_lambda, input_domain, output_domain, nonlinearity, enforce_dc, num_epochs, batch_size in itertools.product(
        datasets, tasks, us_fracs, mot_fracs, max_htranses, max_vtranses, max_rots, noise_stds, architectures, kernel_sizes, num_featureses, num_convses, num_layerses, loss_types, losses, loss_lambdas, input_domains, output_domains, nonlinearities, enforce_dcs, num_epochses, batch_sizes):
    base_dir = os.path.join(filepaths.CONFIG_DIR, exp_name)
    ini_filename = dataset
    for name in [
            task,
            us_frac,
            mot_frac,
            max_htrans,
            max_vtrans,
            max_rot,
            noise_std,
            architecture,
            kernel_size,
            num_features,
            num_convs,
            num_layers,
            loss_type,
            loss,
            loss_lambda,
            input_domain,
            output_domain,
            nonlinearity,
            enforce_dc,
            num_epochs,
            batch_size]:
        ini_filename += '-' + name
    ini_filename += '.ini'

    # Excluded configs
    if(input_domain == output_domain and
       not(input_domain == 'IMAGE' and nonlinearity == '3-piece') and
       not(architecture != 'CONV_RESIDUAL' and num_layers == '20') and
       not(architecture == 'CONV_RESIDUAL' and num_layers == '10') and
       not(input_domain == 'FREQ' and nonlinearity == 'relu') and
       not(input_domain == 'IMAGE' and architecture != 'CONV_RESIDUAL') and
       not(input_domain == 'FREQ' and architecture == 'CONV') and
       not(input_domain == 'IMAGE' and enforce_dc == 'True')):
        dest_file = os.path.join(base_dir, ini_filename)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        f = open(dest_file, "w")
        f.write('[DATA]\n')
        f.write('dataset = ' + dataset + '\n')
        f.write('task = ' + task + '\n')
        f.write('us_frac = ' + us_frac + '\n')
        f.write('mot_frac = ' + mot_frac + '\n')
        f.write('max_htrans = ' + max_htrans + '\n')
        f.write('max_vtrans = ' + max_vtrans + '\n')
        f.write('max_rot = ' + max_rot + '\n')
        f.write('noise_std = ' + noise_std + '\n')

        f.write('\n')

        f.write('[MODEL]\n')
        f.write('architecture = ' + architecture + '\n')
        f.write('kernel_size = ' + kernel_size + '\n')
        f.write('num_features = ' + num_features + '\n')
        f.write('num_convs = ' + num_convs + '\n')
        f.write('num_layers = ' + num_layers + '\n')
        f.write('loss_type = ' + loss_type + '\n')
        f.write('loss = ' + loss + '\n')
        f.write('loss_lambda = ' + loss_lambda + '\n')
        f.write('input_domain = ' + input_domain + '\n')
        f.write('output_domain = ' + output_domain + '\n')
        f.write('nonlinearity = ' + nonlinearity + '\n')
        f.write('enforce_dc = ' + enforce_dc + '\n')

        f.write('\n')

        f.write('[TRAINING]\n')
        f.write('num_epochs = ' + num_epochs + '\n')
        f.write('batch_size = ' + batch_size)
        f.close()
