import argparse
import os
import re
import sys

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import fashion_mnist

import visualization_lib
from interlacer import data_generator, fastmri_data_generator, layers, losses, models, utils
from scripts import filepaths, training_config

matplotlib.use('Agg')

def load_models(exp_folder, selected_exps, epoch=None):
    models = []
    configs = []
    for exp in selected_exps:
        model_path = os.path.join(exp_folder, exp)
        if(epoch is None):
            ckpt = str(visualization_lib.get_best_ckpt(model_path)).zfill(4)
        else:
            ckpt = str(epoch).zfill(4)
        ckptname = 'cp-' + ckpt + '.' + 'ckpt'
    
        config_file = [i for i in os.listdir(
            model_path) if i.endswith('_config.ini')][0]
        config_path = os.path.join(model_path, config_file)
        config, model = visualization_lib.load_model(
            config_path)
        
        if(config.task=='motion' and config.input_domain=='IMAGE'):
            ckpt = str(5).zfill(4)
            ckptname = 'cp-' + ckpt + '.' + 'ckpt'

        exp_config = training_config.TrainingConfig(config_path)
        exp_config.read_config()

        architecture = exp_config.architecture
        input_domain = exp_config.input_domain
        loss = exp_config.loss
        nonlinearity = exp_config.nonlinearity
        num_layers = str(exp_config.num_layers)

        used_loss = losses.image_loss(config.output_domain, config.loss)
        fourier_l1 = losses.fourier_loss(config.output_domain, 'L1')
        fourier_l2 = losses.fourier_loss(config.output_domain, 'L2')
        image_l1 = losses.image_loss(config.output_domain, 'L1')
        image_l2 = losses.image_loss(config.output_domain, 'L2')
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=used_loss,
                      metrics=[fourier_l1, fourier_l2, image_l1, image_l2])
        model.load_weights(os.path.join(model_path, ckptname))

        models.append(model)
        configs.append(exp_config)
    return models, configs

            
def show_image(axis, image, compressed, title=None, ylabel=None, lpips=False, *cmap):
    axis.grid(b=True, axis='both', color='w', linestyle='-', linewidth=2)
    if(lpips):
        print(np.max(image)/ti_max)
        axis.imshow(image, cmap='gray',vmin=ti_min, vmax=500*ti_max)
    else:
        axis.imshow(image, cmap='gray', vmin=ti_min, vmax=ti_max)
    
    rect = patches.Rectangle((coord_x, coord_y), w, h,
                             linewidth=1, edgecolor='r', facecolor='none')

    if(not compressed):
        axis.add_patch(rect)
    if(title is not None):
        axis.set_title(title)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=20)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])


def show_patch(axis, image, title=None, ylabel=None, *cmap):
    axis.grid(b=True, axis='both', color='w', linestyle='-', linewidth=2)


    axis.imshow(
        np.abs(image)[
            coord_y:coord_y + w,
            coord_x:coord_x + w],
        cmap='gray', vmin=ti_min, vmax=ti_max)
    if(title is not None):
        axis.set_title(title)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=20)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])


def show_patch_diff(axis, image, title=None, ylabel=None, *cmap):
    axis.grid(b=True, axis='both', color='w', linestyle='-', linewidth=2)
    
    axis.imshow(
        (
            image -
            true_img)[
            coord_y:coord_y +
            w,
            coord_x:coord_x +
            w],
        cmap='seismic',vmin=-ti_max,vmax=ti_max)
    if(title is not None):
        axis.set_title(title)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=20)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])


def show_mask(axis, image, title=None, ylabel=None, *cmap):
    axis.grid(b=True, axis='both', color='w', linestyle='-', linewidth=2)
    axis.imshow(image, cmap='gray')
    if(title is not None):
        axis.set_title(title)
    axis.axis('off')


def show_FFT(axis, image, ylabel=None, *cmap):
    to_show = image
    axis.imshow(np.fft.fftshift(to_show), cmap='gray', vmin=0, vmax=10)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=20)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])

def show_FFT_diff(axis, image, ylabel=None, *cmap):
    gt_fft = np.log(np.abs(utils.join_reim(m_label)[ind,:,:]))
    to_show = image
    axis.imshow(np.fft.fftshift(np.abs(to_show-gt_fft)), cmap='gray', vmin=0, vmax=10)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=20)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    
def get_label(architecture, input_domain):
    if(architecture == 'CONV_RESIDUAL'):
        if(input_domain == 'FREQ'):
            return 'Frequency'
        elif(input_domain == 'IMAGE'):
            return 'Image'
    if(architecture == 'CONV'):
        if(input_domain == 'IMAGE'):
            return 'IMAGE'
    elif(architecture == 'UNET'):
        return 'UNET'
    elif(architecture == 'INTERLACER_RESIDUAL'):
        return 'Interleaved'
    elif(architecture == 'DOUBLE_INTERLACER_RESIDUAL'):
        return 'DOUBLE JOINT'
    elif(architecture == 'ALTERNATING_RESIDUAL'):
        return 'Alternating'

def get_loss_label(loss):
    if(loss == 'image'):
        return 'Image L1'
    elif(loss == 'freq'):
        return 'Freq L1'
    elif(loss == 'joint'):
        return 'Joint L1'
    elif(loss == 'ssim'):
        return'SSIM'
    elif(loss == 'ssim_ms'):
        return 'Multiscale SSIM'
    elif(loss == 'psnr'):
        return 'PSNR'
    elif(loss == 'lpips'):
        return 'LPIPS'
    
    

def plot_image(model, config, axes, index, img_ind, compressed, lpips=False):
    if(config.output_domain == 'FREQ'):

        model_output_t = model.predict(
            tf.convert_to_tensor(
                m_input, 'float'), steps=1)          

        model_output = np.asarray(model_output_t)
        
        model_output_img_t = tf.expand_dims(
            utils.convert_tensor_to_image_domain(model_output)[
                img_ind, :, :, :], 0)
        model_output_img = np.abs(
            utils.join_reim(
               utils.convert_to_image_domain(model_output)))[
            img_ind, :, :]
        loss_value = str(model.evaluate(tf.expand_dims(tf.convert_to_tensor(
            m_input[img_ind, :, :, :], 'float'), 0), tf.expand_dims(m_label[img_ind, :, :, :], 0))[0])

    elif(config.output_domain == 'IMAGE'):
        model_input_t = tf.convert_to_tensor(
            utils.convert_to_image_domain((m_input)), 'float')
        model_output_t = model.predict(model_input_t, steps=1)
        model_output = np.asarray(model_output_t)

        model_output_img_t = tf.expand_dims(
            model_output_t[img_ind, :, :, :], 0)
        model_output_img = np.abs(
            utils.join_reim(model_output))[
            img_ind, :, :]
        m_label_img = tf.convert_to_tensor(
            utils.convert_to_image_domain(m_label))
        loss_value = str(model.evaluate(tf.expand_dims(tf.convert_to_tensor(model_input_t, 'float')[
                         img_ind, :, :, :], 0), tf.expand_dims(m_label_img[img_ind, :, :, :], 0))[0])

    if(config.output_domain == 'FREQ'):
        model_output_fourier = np.abs(utils.join_reim(model_output))[
            img_ind, :, :]
    elif(config.output_domain == 'IMAGE'):
        model_output_fourier = np.abs(
            utils.join_reim(
                (utils.convert_to_frequency_domain(model_output))))[
            img_ind, :, :]

    label = get_label(config.architecture, config.input_domain)

    ax = axes[0][index]
    show_image(ax, model_output_img, compressed, label, lpips=lpips)

    if(not compressed):
        ax = axes[1][index]
        show_patch(ax, model_output_img)

        ax = axes[2][index]
        show_patch_diff(ax, model_output_img)

        ax = axes[3][index]
        show_FFT(ax, np.log(model_output_fourier))

        ax = axes[4][index]
        show_FFT_diff(ax, np.log(model_output_fourier))
    
    if(compressed):
        ax = axes[1][index]
        show_FFT(ax, np.log(model_output_fourier))


parser = argparse.ArgumentParser(
    description='Compare outputs of neural networks in both the Fourier and image space.')
parser.add_argument(
    'experiment',
    help='Path to directory containing multiple configs.')
parser.add_argument('dataset', help='Dataset sample to run.')
parser.add_argument('corruption_frac', help='Corruption parameter.')
parser.add_argument('--epoch', help='Training epoch to plot.')
parser.add_argument('--compressed', help='Whether to generate smaller version of plot', nargs='?', default=False, const=True)

args = parser.parse_args()
dataset = args.dataset
epoch = args.epoch
compressed = args.compressed
if(epoch is not None):
    epoch = int(epoch)
exp_folder = args.experiment
corruption_frac = str(args.corruption_frac)
exps = sorted(os.listdir(exp_folder))

coord_x, coord_y = [130,120]
w, h = [64, 64]

selected_exps_alph = [exp for exp in exps if (exp[-3:] != 'png' and (exp[-3:] != 'pdf') and (
    '5-piece' not in exp) and ('ipynb' not in exp) and (corruption_frac in exp))]

selected_exps = selected_exps_alph
selected_exps = [selected_exps_alph[i] for i in [1,2,3,0]]

np.random.seed(0)

corruption_frac = float(corruption_frac)
num_models = len(selected_exps)

base_dir = filepaths.FASTMRI_DATA_DIR
test_slice_dir = os.path.join(base_dir, 'validate/singlecoil_val')

exp = selected_exps[0]
if(dataset=='brain'):
    img_test = data_generator.get_mri_TEST_images()

if(dataset=='brain'):
    if('motion' in exp):
        val_generator = data_generator.generate_data(
            img_test, 'motion', 'FREQ', 'FREQ', corruption_frac, split='test', batch_size=8)
    elif('noise' in exp):
        val_generator = data_generator.generate_data(
            img_test, 'noise', 'FREQ', 'FREQ', corruption_frac, batch_size=10)
    else:
        val_generator = data_generator.generate_data(
            img_test, 'undersample', 'FREQ', 'FREQ', corruption_frac, batch_size=10)
elif(dataset=='knee'):
    val_generator = fastmri_data_generator.generate_data(
    test_slice_dir, 'undersample', 'FREQ', 'FREQ', corruption_frac, batch_size=10)

m_input, m_label = next(val_generator)

for ind in range(10):
    true_img_t = tf.expand_dims(
        utils.convert_tensor_to_image_domain(m_label)[
            ind, :, :, :], 0)
    true_img = np.abs(
        utils.join_reim(
            utils.convert_to_image_domain(m_label)))[
        ind, :, :]
    ti_max = np.max(true_img)
    ti_min = np.min(true_img)
    
    input_img = np.abs(
        utils.join_reim(
            utils.convert_to_image_domain(m_input)))[
        ind, :, :]

    true_fourier = np.abs(utils.join_reim(m_label))[ind, :, :]
    input_fourier = np.abs(utils.join_reim(m_input))[ind, :, :]

    if(compressed):
        fig, a = plt.subplots(
            2, 2 + num_models, figsize=(4 * (2 + num_models), 8))
    else:
        fig, a = plt.subplots(
            5, 2 + num_models, figsize=(4 * (2 + num_models), 20))        
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    plt.rcParams.update({'font.size': 16})

    ax = a[0][0]
    show_image(ax, true_img, compressed, 'Ground Truth',  ylabel='Magnitude Image')

    ax = a[0][1]
    show_image(ax, input_img, compressed, 'Input')

    if(not compressed):
        ax = a[1][0]
        show_patch(ax, true_img, ylabel='Image Patch')

        ax = a[1][1]
        show_patch(ax, input_img)        
        
        ax = a[2][0]
        show_patch_diff(ax, true_img, ylabel='Patch Difference')

        ax = a[2][1]
        show_patch_diff(ax, input_img)        
        
        ax = a[3][0]
        show_mask = true_fourier
        show_FFT(ax, np.log(true_fourier), ylabel='Frequency Spectrum') 

        ax = a[3][1]
        show_mask = input_fourier
        show_mask[show_mask == 0] = 1e-4
        show_FFT(ax, np.log(show_mask)) 
        
        ax = a[4][0]
        show_mask = true_fourier
        show_FFT_diff(ax, np.log(true_fourier), ylabel='Frequency Diff') 

        ax = a[4][1]
        show_mask = input_fourier
        show_mask[show_mask == 0] = 1e-4
        show_FFT_diff(ax, np.log(show_mask)) 
        
    else:
        ax = a[1][1]
        show_mask = input_fourier
        show_mask[show_mask == 0] = 1e-4
        show_FFT(ax, np.log(show_mask))

        ax = a[1][0]
        show_mask = true_fourier
        show_FFT(ax, np.log(true_fourier), ylabel='Frequency Spectrum')        

    models, configs = load_models(exp_folder, selected_exps, epoch)


    for i in range(len(models)):
        lpips = ('lpips' in configs[i].loss_type)                             
        plot_image(models[i], configs[i], a, i + 2, ind, compressed, lpips=lpips)
        

    filestr = 'output_comparison_' + str(corruption_frac) + '_' + str(ind)

    if(compressed):
        filestr = 'output_comparison_compressed_' + str(corruption_frac) + '_' + str(ind)
    if(epoch is not None):
        filestr += '_ep'+str(epoch).zfill(4)
    filestr += '.pdf'
    out_dir = os.path.join(
        exp_folder,
        filestr)
    plt.savefig(out_dir, bbox_inches='tight', pad_inches=0.2)
