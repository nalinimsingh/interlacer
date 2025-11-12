import torch

def join_reim(array):
    """Join real and imaginary channels into a complex-valued matrix.

    Args:
        array (torch.Tensor): Real tensor of shape (B, 2, H, W)

    Returns:
        torch.Tensor: Complex-valued tensor of shape (B, H, W)
    """
    assert array.shape[1] == 2, "Input must have 2 channels for real and imaginary parts"
    return torch.complex(array[:, 0, :, :], array[:, 1, :, :])

# Alias for compatibility
join_reim_tensor = join_reim

def join_reim_channels(array):
    """Join real and imaginary channels into a complex tensor.

    Args:
        array (torch.Tensor): Real tensor of shape (B, C, H, W), where C is even (real + imag channels)

    Returns:
        torch.Tensor: Complex tensor of shape (B, C/2, H, W)
    """
    ch = array.shape[1]
    assert ch % 2 == 0, "Channel dimension must be even (real + imaginary)"
    real = array[:, :ch // 2, :, :]
    imag = array[:, ch // 2:, :, :]
    return torch.complex(real, imag)

def split_reim(array):
    """Split a complex-valued tensor into real and imaginary channels.
    
    If input is real-valued, creates a tensor with zeros for imaginary part.

    Args:
        array (torch.Tensor): Complex tensor of shape (B, H, W) or real tensor of shape (B, H, W)

    Returns:
        torch.Tensor: Real tensor of shape (B, 2, H, W)
    """
    if torch.is_complex(array):
        return torch.stack((array.real, array.imag), dim=1)
    else:
        # Real-valued input, add zero imaginary part
        zeros = torch.zeros_like(array)
        return torch.stack((array, zeros), dim=1)

def split_reim_tensor(array):
    """Split a complex tensor into real and imaginary channels.

    Args:
        array (torch.Tensor): Complex tensor of shape (B, H, W)

    Returns:
        torch.Tensor: Real tensor of shape (B, 2, H, W)
    """
    return split_reim(array)

def split_reim_channels(array):
    """Split a complex-valued tensor into real and imaginary channels.

    Args:
        array (torch.Tensor): Complex tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor: Real tensor of shape (B, 2*C, H, W)
    """
    return torch.cat((array.real, array.imag), dim=1)

def convert_to_frequency_domain(images):
    """Convert spatial domain images to frequency domain using FFT.

    Args:
        images (torch.Tensor): Real tensor of shape (B, 2, H, W)

    Returns:
        torch.Tensor: Real tensor of shape (B, 2, H, W) representing FFT
    """
    assert images.shape[1] == 2, "Input must have 2 channels for real and imaginary parts"
    joined = join_reim(images)
    fft = torch.fft.fft2(joined)
    shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    return split_reim(shifted)

# Alias
convert_tensor_to_frequency_domain = convert_to_frequency_domain

def convert_to_image_domain(spectra):
    """Convert frequency domain data back to spatial domain using IFFT.

    Args:
        spectra (torch.Tensor): Real tensor of shape (B, 2, H, W)

    Returns:
        torch.Tensor: Real tensor of shape (B, 2, H, W)
    """
    assert spectra.shape[1] == 2, "Input must have 2 channels for real and imaginary parts"
    joined = join_reim(spectra)
    shifted = torch.fft.ifftshift(joined, dim=(-2, -1))
    img = torch.fft.ifft2(shifted)
    return split_reim(img)

# Alias
convert_tensor_to_image_domain = convert_to_image_domain

def convert_channels_to_freq(images):
    """Convert multi-channel spatial images to frequency domain using FFT.

    Args:
        images (torch.Tensor): Real tensor of shape (B, C, H, W), where C is even

    Returns:
        torch.Tensor: Real tensor of shape (B, C, H, W)
    """
    ch = images.shape[1]
    assert ch % 2 == 0, "Input must have even number of channels (real + imaginary)"
    complex_images = join_reim_channels(images)
    fft = torch.fft.fft2(complex_images)
    split = split_reim_channels(fft)
    return torch.fft.fftshift(split, dim=(-2, -1))

def convert_channels_to_image(spectra):
    """Convert multi-channel frequency domain data back to spatial domain using IFFT.

    Args:
        spectra (torch.Tensor): Real tensor of shape (B, C, H, W), where C is even

    Returns:
        torch.Tensor: Real tensor of shape (B, C, H, W)
    """
    ch = spectra.shape[1]
    assert ch % 2 == 0, "Input must have even number of channels (real + imaginary)"
    shifted = torch.fft.fftshift(spectra, dim=(-2, -1))
    complex_spectra = join_reim_channels(shifted)
    img = torch.fft.ifft2(complex_spectra)
    return split_reim_channels(img)
