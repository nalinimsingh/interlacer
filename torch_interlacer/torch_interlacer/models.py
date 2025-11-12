import torch
import torch.nn as nn

# Custom Package Classes 
from . import layers
from . import utils

# Models Class Implementation in PyTorch
class InterlacerResidualModel(nn.Module):
  def __init__(
    self,
    input_size,
    nonlinearity,
    kernel_size,
    num_features,
    num_convs,
    num_layers,
    enforce_dc=False
  ):
    super().__init__()
    self.input_size = input_size
    self.num_features = num_features
    self.num_layers = num_layers
    layers_l = []
    layers_l.append(layers.Interleaved(
      num_features, kernel_size, num_convs,
      shift=False, hyp_conv=False, in_features=int(input_size[0])
    ))
    for i in range(1, num_layers):
      layers_l.append(layers.Interleaved(
        num_features, kernel_size, num_convs,
        shift=False, hyp_conv=False, in_features=num_features
      ))
    self.layers = nn.ModuleList(layers_l)
    self.out_conv2d = nn.Conv2d(num_features, 2, kernel_size, padding='same', bias=True)
    nn.init.kaiming_normal_(self.out_conv2d.weight, mode='fan_in', nonlinearity='linear')
    if self.out_conv2d.bias is not None:
      nn.init.zeros_(self.out_conv2d.bias)
    self.enforce_dc = enforce_dc
  
  def forward(self, x, dc_mask=None):
    """
    x: (B, 2, H, W)  —  real & imag
    dc_mask: optional (B,1,H,W) of 0/1 if you want to keep DC untouched
    """
    B, C, H, W = x.shape
    assert C == 2, "expect exactly 2 channels (real+imag)"

    # --- build the frequency-domain “residual” tile ---
    # this takes 2 channels and repeats each one n_copies times:
    # result is (B, 2*n_copies, H, W) == (B, num_features, H, W)
    n_copies = self.num_features // 2
    inp_copy = x.repeat_interleave(n_copies, dim=1)
    
    inputs_img   = utils.convert_channels_to_image(x)
    inp_img_copy = inputs_img.repeat_interleave(n_copies, dim=1)
    
    freq_in, img_in = x, inputs_img
    # pass through interleaved blocks
    for layer in self.layers:
      img_conv, k_conv = layer((img_in, freq_in))
      freq_in = k_conv + inp_copy
      img_in  = img_conv + inp_img_copy
    out = self.out_conv2d(freq_in) + x
    # optional DC enforcement
    if self.enforce_dc:
      assert dc_mask is not None, "need dc_mask when enforce_dc=True"
      out = dc_mask * x + (1 - dc_mask) * out
    return out
  
class ConvNoResidualModel(nn.Module):
    def __init__(self, input_size, nonlinearity, kernel_size, num_features, num_layers, enforce_dc=False):
        super().__init__()
        self.enforce_dc = enforce_dc
        self.input_channels = input_size[0]
        conv_layers = []

        in_channels = self.input_channels
        for _ in range(num_layers):
            conv_layers.append(layers.BatchNormConv(in_channels, num_features, kernel_size))
            conv_layers.append(layers.get_nonlinear_layer(nonlinearity))
            in_channels = num_features

        self.conv_stack = nn.Sequential(*conv_layers)
        self.out_conv = nn.Conv2d(num_features, 2, kernel_size, padding='same', bias=False)
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x, dc_mask=None):
        out = self.conv_stack(x)
        out = self.out_conv(out)
        if self.enforce_dc:
            assert dc_mask is not None
            out = dc_mask * x + (1 - dc_mask) * out
        return out
      
class ConvResidualModel(nn.Module):
    def __init__(self, input_size, nonlinearity, kernel_size, num_features, num_layers, enforce_dc=False):
        super().__init__()
        self.enforce_dc = enforce_dc
        self.input_channels = input_size[0]
        self.n_copies = num_features // 2

        conv_layers = []
        in_channels = self.input_channels

        for _ in range(num_layers):
            conv_layers.append(layers.BatchNormConv(in_channels, num_features, kernel_size))
            conv_layers.append(layers.get_nonlinear_layer(nonlinearity))
            in_channels = num_features

        self.conv_stack = nn.Sequential(*conv_layers)
        self.out_conv = nn.Conv2d(num_features, 2, kernel_size, padding='same', bias=False)
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x, dc_mask=None):
        B, C, H, W = x.shape
        inp_copy = x.repeat_interleave(self.n_copies, dim=1)

        out = x
        for i in range(0, len(self.conv_stack), 2):
            out = self.conv_stack[i](out)
            out = self.conv_stack[i+1](out)
            out = out + inp_copy

        out = self.out_conv(out)
        out = out + x
        if self.enforce_dc:
            assert dc_mask is not None
            out = dc_mask * x + (1 - dc_mask) * out
        return out