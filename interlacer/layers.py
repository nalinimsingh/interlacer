import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import utils


def piecewise_relu(x):
    """Custom nonlinearity for freq-space convolutions."""
    return x + keras.activations.relu(1 / 2 * (x - 1)) + \
        keras.activations.relu(1 / 2 * (-1 - x))


get_custom_objects().update({'piecewise_relu': Activation(piecewise_relu)})


def get_nonlinear_layer(nonlinearity):
    """Selects and returns an appropriate nonlinearity."""
    if(nonlinearity == 'relu'):
        return tf.keras.layers.Lambda(keras.activations.relu)
    elif(nonlinearity == '3-piece'):
        return tf.keras.layers.Lambda(Activation(piecewise_relu))


class BatchNormConv(Layer):
    """Custom layer that combines BN and a convolution."""

    def __init__(self, features, kernel_size, complex_conv, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        self.complex_conv = complex_conv
        super(BatchNormConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization()
        self.full_conv = Conv2D(
            self.features,
            self.kernel_size,
            activation=None,
            padding='same',
            kernel_initializer='he_normal')

        self.real_conv = Conv2D(
            int(self.features//2),
            self.kernel_size,
            activation=None,
            padding='same',
            kernel_initializer='he_normal')

        self.imaginary_conv = Conv2D(
            self.features//2,
            self.kernel_size,
            activation=None,
            padding='same',
            kernel_initializer='he_normal')

        super(BatchNormConv, self).build(input_shape)

    def call(self, x):
        """Core layer function to combine BN/convolution.

        Args:
          x: Input tensor

        Returns:
          conv(float): Output of BN (on axis 0) followed by convolution

        """
        bn = self.bn(x)

        if(self.complex_conv):
            num_channels = bn.shape[-1]
            real_half = bn[:,:,:,0:num_channels//2]
            imaginary_half = bn[:,:,:,num_channels//2:]

            IRxKR = self.real_conv(real_half)
            IIxKI = self.imaginary_conv(imaginary_half)
            IRxKI = self.imaginary_conv(real_half)
            IIxKR = self.real_conv(imaginary_half)

            real_FM = tf.subtract(IRxKR,IIxKI)
            imaginary_FM = tf.add(IRxKI,IIxKR)

            conv = tf.concat([real_FM, imaginary_FM], axis=3)

        else:
            conv = self.full_conv(bn)

        return conv

    def compute_output_shape(self, input_shape):
        return (input_shape[:3] + [self.features])


class Mix(Layer):
    """Custom layer to learn a combination of two inputs."""

    def __init__(self, **kwargs):
        super(Mix, self).__init__(**kwargs)

    def build(self, input_shape):
        self._mix = self.add_weight(name='mix_param',
                                  shape=(1,),
                                  initializer='uniform',
                                  trainable=True)
        super(Mix, self).build(input_shape)

    def call(self, x):
        """Core layer function to combine inputs.

        Args:
          x: Tuple (A,B), where A and B are numpy arrays of equal shape

        Returns:
          sig_mix*A + (1-sig_mix)B, where six_mix = sigmoid(mix) and mix is a learned combination parameter

        """
        A, B = x
        sig_mix = tf.math.sigmoid(self._mix)
        return sig_mix * A + (1 - sig_mix) * B

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Interlacer(Layer):
    """Custom layer to learn features in both image and frequency space."""

    def __init__(self, features, kernel_size, complex_conv, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        self.complex_conv = complex_conv
        super(Interlacer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.img_mix = Mix()
        self.freq_mix = Mix()
        self.img_bnconv = BatchNormConv(self.features, self.kernel_size, self.complex_conv)
        self.freq_bnconv = BatchNormConv(self.features, self.kernel_size, self.complex_conv)
        super(Interlacer, self).build(input_shape)

    def call(self, x):
        """Core layer function to learn image and frequency features.

        Args:
          x: Tuple (A,B), where A contains image-space features and B contains frequency-space features

        Returns:
          img_conv(float): nonlinear(conv(BN(beta*img_in+IFFT(freq_in))))
          freq_conv(float): nonlinear(conv(BN(alpha*freq_in+FFT(img_in))))

        """
        img_in, freq_in = x

        img_in_as_freq = utils.convert_channels_to_frequency_domain(img_in)
        freq_in_as_img = utils.convert_channels_to_image_domain(freq_in)

        mixed_ilayer_input = self.img_mix([img_in, freq_in_as_img])
        mixed_flayer_input = self.freq_mix([freq_in, img_in_as_freq])

        img_conv = self.img_bnconv(mixed_ilayer_input)
        img_nonlinear = get_nonlinear_layer('relu')(img_conv)

        k_conv = self.freq_bnconv(mixed_flayer_input)
        k_nonlinear = get_nonlinear_layer('3-piece')(k_conv)

        return (img_nonlinear, k_nonlinear)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][:3] + [self.features],
                input_shape[1][:3] + [self.features])
