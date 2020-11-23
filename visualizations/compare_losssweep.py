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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    
        print(model_path)
        print([i for i in os.listdir(model_path) if i.endswith('_config.ini')])
        config_file = [i for i in os.listdir(
            model_path) if i.endswith('_config.ini')][0]
        config_path = os.path.join(model_path, config_file)
        config, model = visualization_lib.load_model(
            config_path)

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
        print('loaded model')
        models.append(model)
        configs.append(exp_config)
    return models, configs

            
def show_image(axis, image, title=None, ylabel=None, *cmap):
    axis.grid(b=True, axis='both', color='w', linestyle='-', linewidth=2)
    axis.imshow(image, cmap='gray', vmin=ti_min, vmax=ti_max)

    if(title is not None):
        axis.set_title(title, fontsize=25)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=25, rotation=90)
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
    
def get_arch_row(arch_label):
    if(arch_label=='Image'):
        return 0
    elif(arch_label=='Frequency'):
        return 1
    elif(arch_label=='Alternating'):
        return 2
    elif(arch_label=='Interleaved'):
        return 3
    

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
    
def get_loss_col(loss_label):
    if(loss_label=='Image L1'):
        return 2
    elif(loss_label=='Freq L1'):
        return 3
    elif(loss_label=='Joint L1'):
        return 4
    elif(loss_label=='SSIM'):
        return 5
    elif(loss_label=='Multiscale SSIM'):
        return 6
    elif(loss_label=='PSNR'):
        return 7

def plot_inset(ax, img):
    axins = inset_axes(ax, width=1, height=1)
    axins.imshow(img[110:210,20:120], cmap='gray', vmin=ti_min, vmax=ti_max)

    axins.get_xaxis().set_ticks([])
    axins.get_yaxis().set_ticks([])

    for pos in ['top', 'bottom', 'right', 'left']:
        axins.spines[pos].set_edgecolor('white')    
    
    
def plot_image(model, config, axes, index, img_ind):
    if(config.output_domain == 'FREQ'):

        if(config.architecture!='DOUBLE_INTERLACER_RESIDUAL'):
            model_output_t = model.predict(
                tf.convert_to_tensor(
                    m_input, 'float'), steps=1)
        else:
            [model_output_t, _] = model.predict(
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
            utils.convert_to_image_domain(np.fft.ifftshift(m_label,axes=(1,2))))
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

    loss_label = get_loss_label(config.loss_type)
    arch_label = get_label(config.architecture, config.input_domain)

    if(get_loss_col(loss_label) is not None):
        ax = axes[(get_arch_row(arch_label))*3][get_loss_col(loss_label)]
        ax.yaxis.set_label_position("right")
        if(get_loss_col(loss_label)==7):
            ylabel=arch_label
        else:
            ylabel=None
            
        if(get_arch_row(arch_label)==0):
            title=loss_label
        else:
            title=None

        show_image(ax, model_output_img, title=title, ylabel=ylabel)
        rect = patches.Rectangle((20, 110), 100, 100,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
        ax = axes[(get_arch_row(arch_label))*3+1][get_loss_col(loss_label)]
        ax.yaxis.set_label_position("right")
        if(ylabel is not None):
            ylabel=ylabel+' (Patch)'
        show_image(ax, model_output_img[110:210,20:120], title=None, ylabel=ylabel)


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

selected_exps_alph = [exp for exp in exps if (exp[-3:] != 'pdf'  and (exp[-3:] != 'png') and (
    '5-piece' not in exp) and ('ipynb' not in exp) and (corruption_frac in exp))]

selected_exps = selected_exps_alph

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
            img_test, 'noise', 'FREQ', 'FREQ', corruption_frac, batch_size=5)
    else:
        val_generator = data_generator.generate_data(
            img_test, 'undersample', 'FREQ', 'FREQ', corruption_frac, batch_size=5)
elif(dataset=='knee'):
    val_generator = fastmri_data_generator.generate_data(
    test_slice_dir, 'undersample', 'FREQ', 'FREQ', corruption_frac, batch_size=5)

m_input, m_label = next(val_generator)

for ind in range(3):
    true_img_t = tf.expand_dims(
        utils.convert_tensor_to_image_domain(m_label)[
            ind, :, :, :], 0)
    true_img = np.abs(
        utils.join_reim(
            utils.convert_to_image_domain(np.fft.ifftshift(m_label,axes=(1,2)))))[
        ind, :, :]
    ti_max = np.max(true_img)
    ti_min = np.min(true_img)
    
    input_img = np.abs(
        utils.join_reim(
            utils.convert_to_image_domain(np.fft.ifftshift(m_input,axes=(1,2)))))[
        ind, :, :]

    true_fourier = np.abs(utils.join_reim(m_label))[ind, :, :]
    input_fourier = np.abs(utils.join_reim(m_input))[ind, :, :]

    fig, a = plt.subplots(
        12, 8, figsize=(4 * 8, 34),gridspec_kw=dict(height_ratios=[4,4,0.5,4,4,0.5,4,4,0.5,4,4,0.5])) 
    
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    plt.rcParams.update({'font.size': 16})

    ax = a[0][0]
    show_image(ax, true_img, 'Ground Truth')
    rect = patches.Rectangle((20, 110), 100, 100,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ax = a[1][0]
    show_image(ax, true_img[110:210,20:120], title=None, ylabel=None)
    
    ax = a[0][1]
    show_image(ax, input_img, 'Input')
    rect = patches.Rectangle((20, 110), 100, 100,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ax = a[1][1]
    show_image(ax, input_img[110:210,20:120], title=None, ylabel=None)
    
    for i in range(2,12):
        fig.delaxes(a[i][0])
        fig.delaxes(a[i][1])
        if(i%3==2):
            for j in range(2,8):
                fig.delaxes(a[i][j])

    models, configs = load_models(exp_folder, selected_exps, epoch)

    for i in range(len(models)):
        plot_image(models[i], configs[i], a, i, ind)
        
    fig.tight_layout()
    
    filestr = 'losscomparison_images_expanded_' + str(corruption_frac) + '_' + str(ind)
    
    if(epoch is not None):
        filestr += '_ep'+str(epoch).zfill(4)
    filestr += '.pdf'
    out_dir = os.path.join(
        exp_folder,
        filestr)
    plt.savefig(out_dir, bbox_inches='tight', pad_inches=0.2)
