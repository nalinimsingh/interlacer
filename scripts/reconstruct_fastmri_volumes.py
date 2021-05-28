"""Reconstructs the 

  Loads dataset and model from training config information, sets up checkpointing and tensorboard callbacks, and starts training.

  Usage:

    $ python train.py /path/to/config.ini --debug --experiment loss_comparison_runs --suffix trial1

  Options:

    --debug(Boolean): Only train for 5 epochs on limited data, and delete temp logs
    --experiment(string): Optional label for a higher-level directory in which to store this run's log directory
    --suffix(string): Optional, arbitrary tag to append to job name
"""

import argparse
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from interlacer import losses, models, utils
from scripts import training_config, filepaths
from visualizations import visualization_lib


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

        return np.fft.ifftshift(mask)

    else:
        return(np.ones(arr_shape))


def get_fastmri_kspace(image_path):
    """Load and normalize MRI dataset.

    Args:
      image_dir(str): Directory containing 3D MRI volumes of shape (?, n, n); each volume is stored as a '.h5' file in the FastMRI format
      batch_size(int): Number of input-output pairs in each batch
      fs(Boolean): Whether to read images with fat suppression (True) or without (False)

    Returns:
      float: A numpy array of size (num_images, n, n) containing all image slices

    """

    with h5py.File(image_path, "r") as f:
        fs = 'CORPDFS' in f.attrs['acquisition']

        n_slices = f['kspace'].shape[0]

        n = 160
        
        kspace = np.empty((n_slices,2*n,2*n),dtype='complex')
        
        for i in range(n_slices):
            sl_k = f['kspace'][i, :, :]
            mask = f['mask']
            sl_k *= mask

            # Bring k-space down to square size
            sl_k = np.fft.ifftshift(sl_k)

            sl_img_unshift = np.fft.fftshift(np.fft.ifft2(sl_k))
            x_mid = int(sl_img_unshift.shape[0] / 2)
            y_mid = int(sl_img_unshift.shape[1] / 2)
            sl_img_crop = sl_img_unshift[x_mid -
                                         n:x_mid + n, y_mid - n:y_mid + n]
                
            sl_k = np.fft.fft2(sl_img_crop)

            
            if(np.sum(mask)<60):
                acc = '8x'
            else:
                acc = '4x'

            kspace[i,:,:] = sl_k 
            
    return kspace, fs, acc


def reconstruct_vol(path,models):
    corrupt_k, fs, acc = get_fastmri_kspace(path)
    corrupt_k = utils.split_reim(corrupt_k)*500
    
    corrupt_img = utils.convert_to_image_domain(corrupt_k)
    nf = np.percentile(np.abs(corrupt_img),95,axis=(1,2,3))

    n = corrupt_k.shape[1]

    fs_label = 'FS' if fs else 'noFS'
    model_label = acc+'_'+fs_label
    
    model = models[model_label]    
    
    m_output = np.empty(corrupt_k.shape)
    for i in range(corrupt_k.shape[0]):
        corrupt_k[i,:,:,:] /= nf[i]    
        m_output_i = model.predict(np.expand_dims(corrupt_k[i,:,:,:],0))
        m_output[i,:,:,:] = m_output_i
    
    m_output_img = utils.convert_to_image_domain(m_output)
        
    for i in range(corrupt_k.shape[0]):
        m_output_img[i,:,:,:] *= nf[i]
    m_output = np.abs(utils.join_reim(m_output_img))
    
    for i in range(m_output.shape[0]):
        inds = list(range(3))
        inds.extend(list(range(317,320)))
        for j in inds:
            for k in inds:
                m_output[i,j,k] = m_output[i,4,4]
    

    
    return m_output


def load_model(model_path):
    ckpt = str(visualization_lib.get_last_ckpt(model_path)).zfill(4)
    ckptname = 'cp-' + ckpt + '.' + 'ckpt'

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
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Fourier-domain neural network to correct corrupted k-space data.')
    parser.add_argument('vol_path', help='Path to volume to reconstruct.')
    parser.add_argument('recon_path', help='Path to folder with containing model reconstructions')
    parser.add_argument('recon_name', help='Experiment folder name.')
    
    # Set up config
    args = parser.parse_args()
    vol_path = args.vol_path
    recon_path = args.recon_path
    experiment = args.recon_name
    
    experiment_dir = '/data/vision/polina/scratch/nmsingh/dev/fouriernetworks/training/fastmri_leaderboard/'+experiment
    model_paths = [i for i in os.listdir(experiment_dir) if i[-4:]!='.pdf']
    
    non_fs_4x_model_path = os.path.join(experiment_dir,[i for i in model_paths if (('0.75' in i) and ('FS' not in i))][0])
    non_fs_8x_model_path = os.path.join(experiment_dir,[i for i in model_paths if (('0.875' in i) and ('FS' not in i))][0])
    fs_4x_model_path = os.path.join(experiment_dir,[i for i in model_paths if (('0.75' in i) and ('FS' in i))][0])
    fs_8x_model_path = os.path.join(experiment_dir,[i for i in model_paths if (('0.875' in i) and ('FS' in i))][0])
    
    model_paths = {'4x_noFS': non_fs_4x_model_path, 
                   '8x_noFS': non_fs_8x_model_path,
                   '4x_FS': fs_4x_model_path,
                   '8x_FS': fs_8x_model_path
                  }

    
    
    exp_recon_path = os.path.join(recon_path,experiment)
    if not os.path.exists(exp_recon_path):
        os.makedirs(exp_recon_path)
        
    models = {key:load_model(value) for (key,value) in model_paths.items()}
    
    for vol in os.listdir(vol_path):
        
        name = vol
        name_first = name.split('.')[0]
        if(name_first) not in os.listdir(exp_recon_path):        
            recons = reconstruct_vol(os.path.join(vol_path,vol), models)



            with h5py.File(os.path.join(exp_recon_path,name_first+'.h5'), 'w') as hf:
                hf.create_dataset("reconstruction", data=recons)

    
    