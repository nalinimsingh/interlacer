import torch
import torch.nn as nn
# local packages - use conda develop
from . import utils


class PiecewiseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.ReLU()
        
    def forward(self, x):
        return x + self.f((x - 1) / 2) + self.f((-1 - x) / 2)

class BatchNormConv(nn.Module):
    "Performs Batch Normalization followed by a convolution"
    def __init__(self, in_channels, out_channels, kernel_size, hyp_conv=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.hyp_conv = hyp_conv
        if self.hyp_conv:
            self.conv = nn.HyperConv2DFromDense(out_channels, kernel_size, padding='same')
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        if self.hyp_conv:
            x, hyp_tensor = x
            x = self.bn(x)
            x = self.conv((x, hyp_tensor))
        else:
            x = self.bn(x)
            x = self.conv(x)
        return x

class Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix_param = nn.Parameter(torch.rand(1))
        self.s = nn.Sigmoid()

    def forward(self, x):
        A, B = x
        sig_mix = self.s(self.mix_param)
        return sig_mix * A + (1 - sig_mix) * B

class Interleaved(nn.Module):
    def __init__(self, num_features, kernel_size=5, num_convs=1, shift=False, hyp_conv=False, in_features=2):
        super().__init__()
        self.in_features = in_features
        self.features = num_features
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.shift = shift
        self.hyp_conv = hyp_conv
        self.img_mix = Mix()
        self.freq_mix = Mix()
        self.img_bnconvs = nn.ModuleList([BatchNormConv(in_features, num_features, kernel_size=kernel_size, hyp_conv=hyp_conv)] + [BatchNormConv(num_features, num_features, kernel_size=kernel_size, hyp_conv=hyp_conv) for i in range(1, num_convs)])
        self.freq_bnconvs = nn.ModuleList([BatchNormConv(in_features, num_features, kernel_size=kernel_size, hyp_conv=hyp_conv)] + [BatchNormConv(num_features, num_features, kernel_size=kernel_size, hyp_conv=hyp_conv) for i in range(1, num_convs)])
        self.relu = nn.ReLU()
        self.p_relu = PiecewiseReLU()
        
    def forward(self, x):
        if self.hyp_conv:
            img_in, freq_in, hyp_tensor = x
        else:
            img_in, freq_in = x
        img_in_as_freq = utils.convert_channels_to_freq(img_in)
        freq_in_as_img = utils.convert_channels_to_image(freq_in)

        img_feat = self.img_mix([img_in, freq_in_as_img])
        k_feat = self.freq_mix([freq_in, img_in_as_freq])
        for i in range(self.num_convs):
            # process image-space features
            img_bn = {}
            if self.shift:
                img_feat = torch.fft.ifftshift(img_feat, dim=(2,3))
            if self.hyp_conv:
                img_conv = self.img_bnconvs[i]((img_feat, hyp_tensor))
            else:
                img_conv = self.img_bnconvs[i](img_feat)
            img_feat = self.relu(img_conv)
            # process frequency-space features
            k_bn = {}
            if self.shift:
                k_feat = torch.fft.ifftshift(k_feat, dim=(2,3))
            if self.hyp_conv:
                k_conv = self.freq_bnconvs[i]((k_feat, hyp_tensor))
            else:
                k_conv = self.freq_bnconvs[i](k_feat)
            k_feat = self.p_relu(k_conv)
        return (img_feat, k_feat)


def get_nonlinear_layer(nonlinearity):
    """Selects and returns an appropriate nonlinearity layer.
    
    Args:
        nonlinearity (str): 'relu' or '3-piece'
        
    Returns:
        nn.Module: The appropriate nonlinearity layer
    """
    if nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == '3-piece':
        return PiecewiseReLU()
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")