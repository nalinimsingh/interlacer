import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom


class UndersampledMRIDataset(Dataset):
    """Dataset for MRI slices with undersampling."""
    
    def __init__(self, images, us_frac=0.75, input_domain='IMAGE', output_domain='IMAGE', transform=None, target_size=256):
        self.images = images
        self.us_frac = us_frac
        self.input_domain = input_domain
        self.output_domain = output_domain
        self.transform = transform
        self.target_size = target_size
        
        # Create undersampling mask for target size
        self.mask = self._create_undersampling_mask(target_size, us_frac)
        
    def _create_undersampling_mask(self, size, us_frac):
        """Create center-preserving undersampling mask."""
        mask = np.zeros((size, size), dtype=bool)
        band_size = 40
        center = size // 2
        keep_band = (center - band_size//2, center + band_size//2)
        
        # Keep center band
        mask[keep_band[0]:keep_band[1]+1, :] = True
        
        # Calculate how many extra lines to keep
        total_lines = size
        center_lines = keep_band[1] - keep_band[0] + 1
        target_lines = int((1 - us_frac) * total_lines)
        extra_lines = max(0, target_lines - center_lines)
        
        if extra_lines > 0:
            # Randomly select extra lines from outside the center band
            available_lines = (keep_band[0] - 0) + (size - keep_band[1] - 1)
            if available_lines > 0:
                extra_per_side = extra_lines // 2
                lines_to_keep = np.random.choice(available_lines, min(extra_per_side, available_lines), replace=False)
                
                # Add lines above center band
                above_lines = lines_to_keep[lines_to_keep < keep_band[0]]
                mask[above_lines, :] = True
                
                # Add lines below center band  
                below_lines = lines_to_keep[lines_to_keep >= keep_band[0]] - keep_band[0] + keep_band[1] + 1
                below_lines = below_lines[below_lines < size]
                mask[below_lines, :] = True
        
        return mask
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]  # Original image data
        img = img.T
        # Convert to complex and take FFT to simulate k-space acquisition
        img_complex = img.astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(img_complex))
        
        # Apply undersampling mask
        kspace_undersampled = kspace * self.mask
        
        # Convert back to image domain (corrupted image from undersampled k-space)
        img_undersampled = np.fft.ifft2(np.fft.ifftshift(kspace_undersampled))
        img_undersampled = np.real(img_undersampled)
        
        # Prepare input and output based on domains
        if self.input_domain == 'IMAGE':
            # Input: corrupted image (real + imaginary channels)
            input_data = np.stack([img_undersampled, np.zeros_like(img_undersampled)], axis=0)
        else:  # FREQ
            # Input: undersampled k-space
            input_data = np.stack([np.real(kspace_undersampled), np.imag(kspace_undersampled)], axis=0)
        
        if self.output_domain == 'IMAGE':
            # Output: original clean image
            output_data = np.stack([img, np.zeros_like(img)], axis=0)
        else:  # FREQ
            # Output: original k-space
            output_data = np.stack([np.real(kspace), np.imag(kspace)], axis=0)
        
        return torch.from_numpy(input_data).float(), torch.from_numpy(output_data).float()
    
    
def mri_safe_resize(dataset, target=(256, 256)):
    out_set = []
    for x in dataset:
        if x.shape == target:
            pass
        elif x.shape[0] < target[0]:
            # add vertical padding
            t_pad = (target[0] - x.shape[0]) // 2
            b_pad = t_pad + (target[0] - x.shape[0]) % 2
            # add horizontal padding
            l_pad = (target[1] - x.shape[1]) // 2
            r_pad = l_pad + (target[1] - x.shape[1]) % 2
            # check upper
            x = np.pad(x, pad_width=((t_pad, b_pad), (l_pad, r_pad)))
        else:
            # crop vertical
            t_crop = (x.shape[0] - target[0]) // 2
            b_crop = t_crop + (x.shape[0] - target[0]) % 2
            l_crop = (x.shape[1] - target[1]) // 2
            r_crop = l_crop + (x.shape[1] - target[1]) % 2
            # crop horizontal 
            x = x[t_crop:-b_crop, l_crop:-r_crop]
        out_set.append(x)
    return out_set

def normalize(dataset):
    # since arrays are normalized at this point, we can take mean of means without weights
    s = 0
    mx = dataset[0].flatten()[0]
    for i in range(len(dataset)):
        x = dataset[i]
        s += x.mean()
        cur_mx = x.max()
        if cur_mx > mx:
            mx = cur_mx
    mean = s / len(dataset)
    # normalize
    for i in range(len(dataset)):
        dataset[i] = (dataset[i] - mean)/ mx
    return dataset
        

def preprocess_slices(all_data, test_split=0.2):
    n = len(all_data)
    indices = np.arange(0, n)
    np.random.shuffle(indices)
    test_idxs = set(indices[:int(n * test_split)])
    train_idxs = set(indices[int(n * test_split):])

    test_data = [all_data[i].astype(np.float32) for i in test_idxs]
    train_data = [all_data[i].astype(np.float32) for i in train_idxs]

    test_data = mri_safe_resize(test_data)
    train_data = mri_safe_resize(train_data)

    test_data = normalize(test_data)
    train_data = normalize(train_data)
    return train_data, test_data