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

    def __init__(self, features, kernel_size, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        super(BatchNormConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization()
        self.conv = Conv2D(
            self.features,
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
        conv = self.conv(bn)

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

    def __init__(self, features, kernel_size, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        super(Interlacer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.img_mix = Mix()
        self.freq_mix = Mix()
        self.img_bnconv = BatchNormConv(self.features, self.kernel_size)
        self.freq_bnconv = BatchNormConv(self.features, self.kernel_size)
        self.img_bnconv2 = BatchNormConv(self.features, self.kernel_size)
        self.freq_bnconv2 = BatchNormConv(self.features, self.kernel_size)
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

        img_in_permuted = Permute((3, 1, 2))(utils.join_reim_channels(img_in))
        img_in_as_freq = utils.split_reim_channels(
            Permute((2, 3, 1))(tf.signal.fft2d(img_in_permuted)))

        freq_in_permuted = Permute((3, 1, 2))(
            utils.join_reim_channels(freq_in))
        freq_in_as_img = utils.split_reim_channels(
            Permute((2, 3, 1))(tf.signal.ifft2d(freq_in_permuted)))

        mixed_ilayer_input = self.img_mix([img_in, freq_in_as_img])
        mixed_flayer_input = self.freq_mix([freq_in, img_in_as_freq])

        img_conv = self.img_bnconv(mixed_ilayer_input)
        img_nonlinear = get_nonlinear_layer('relu')(img_conv)

        img_conv = self.img_bnconv2(img_nonlinear)
        img_nonlinear = get_nonlinear_layer('relu')(img_conv)

        k_conv = self.freq_bnconv(mixed_flayer_input)
        k_nonlinear = get_nonlinear_layer('3-piece')(k_conv)

        k_conv = self.freq_bnconv2(k_nonlinear)
        k_nonlinear = get_nonlinear_layer('3-piece')(k_conv)

        return (img_nonlinear, k_nonlinear)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][:3] + [self.features],
                input_shape[1][:3] + [self.features])
