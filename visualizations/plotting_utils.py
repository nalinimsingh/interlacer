import numpy as np
from matplotlib import pyplot as plt

from interlacer import utils

def plot_k(k,i,vmin=None,vmax=None, ax=None):
    if(ax is not None):
        ax.imshow(np.log(np.abs(utils.join_reim(k))+1e-8)[i,:,:],cmap='gray',vmin=vmin,vmax=vmax)
        ax.axis('off')
    else:
        plt.figure()
        plt.imshow(np.log(np.abs(utils.join_reim(k))+1e-8)[i,:,:],cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
    
def plot_img(img,i,vmin=None,vmax=None, ax=None):
    if(ax is not None):
        ax.imshow(np.abs(utils.join_reim(img))[i,:,:],cmap='gray',vmin=vmin,vmax=vmax)
        ax.axis('off')
    else:
        plt.figure()
        plt.imshow(np.abs(utils.join_reim(img))[i,:,:],cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
    
def plot_img_diff(img1,img2,i, vmin=-1, vmax=1, ax=None):
    if(ax is not None):
        ax.imshow((np.abs(utils.join_reim(img1))-np.abs(utils.join_reim(img2)))[i,:,:],cmap='seismic',vmin=vmin,vmax=vmax)
        ax.axis('off')
    else:
        plt.figure()
        plt.imshow((np.abs(utils.join_reim(img1))-np.abs(utils.join_reim(img2)))[i,:,:],cmap='seismic',vmin=vmin,vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        
        
def plot_k_diff(k1,k2,i, ax=None):
    if(ax is not None):
        ax.imshow(np.log(1e-7+np.abs(utils.join_reim(k1)-utils.join_reim(k2)))[i,:,:],cmap='seismic')
        ax.axis('off')
    else:
        plt.figure()
        plt.imshow(np.log(1e-7+np.abs(utils.join_reim(k1)-utils.join_reim(k2)))[i,:,:],cmap='seismic')
        plt.xticks([])
        plt.yticks([])
    
def plot_img_from_k(k,i,vmin=None,vmax=None, ax=None, fftshift=False):
    to_plot = utils.convert_to_image_domain(k)
    if(fftshift):
        to_plot = np.fft.fftshift(to_plot, axes=(1,2))
    plot_img(to_plot,i,vmin=vmin,vmax=vmax, ax=ax)
    
def plot_k_from_img(img,i,vmin=None,vmax=None, ax=None):
    plot_k(utils.convert_to_frequency_domain(img),i,vmin=vmin,vmax=vmax, ax=ax)
    
def plot_img_from_k_diff(k1,k2,i, vmin=-1, vmax=1, ax=None):
    plot_img_diff(utils.convert_to_image_domain(k1), utils.convert_to_image_domain(k2),i, vmin=vmin, vmax=vmax,ax=ax)