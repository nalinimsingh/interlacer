import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import utils
from neurite.tf import layers as nel

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

    def __init__(self, features, kernel_size, hyp_conv=False, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        self.hyp_conv = hyp_conv
        super(BatchNormConv, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'features': self.features,
            'kernel_size': self.kernel_size,
            'hyp_conv': self.hyp_conv})
        return config

    def build(self, input_shape):
        self.bn = BatchNormalization()
        if(self.hyp_conv):
            conv_layer = nel.HyperConv2DFromDense
            kwargs = {}
        else:
            conv_layer = Conv2D
            kwargs = {'kernel_initializer':'he_normal'}

        self.conv = conv_layer(
            self.features,
            self.kernel_size,
            activation=None,
            padding='same',
            **kwargs)
        super(BatchNormConv, self).build(input_shape)

    def call(self, x):
        """Core layer function to combine BN/convolution.

        Args:
          x: Input tensor (or, if HyperConv, a tuple of input tensor and
          hypernetwork output)

        Returns:
          conv(float): Output of BN (on axis 0) followed by convolution

        """
        if(self.hyp_conv):
            bn = self.bn(x[0])
            conv = self.conv((bn, x[1]))
        else:
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

    def __init__(self, features, kernel_size, num_convs=1, shift=False,
            hyp_conv=False, **kwargs):
        self.features = features
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.shift = shift
        self.hyp_conv = hyp_conv
        super(Interlacer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'features': self.features,
            'kernel_size': self.kernel_size,
            'num_convs': self.num_convs,
            'shift': self.shift,
            'hyp_conv': self.hyp_conv})
        return config
        
    def build(self, input_shape):
        self.img_mix = Mix()
        self.freq_mix = Mix()
        self.img_bnconvs = [BatchNormConv(self.features, self.kernel_size,
            hyp_conv=self.hyp_conv) for i in range(self.num_convs)]
        self.freq_bnconvs = [BatchNormConv(self.features, self.kernel_size,
            hyp_conv=self.hyp_conv) for i in range(self.num_convs)]
        super(Interlacer, self).build(input_shape)

    def call(self, x):
        """Core layer function to learn image and frequency features.

        Args:
          x: Tuple (A,B), where A contains image-space features and B contains frequency-space features

        Returns:
          img_conv(float): nonlinear(conv(BN(beta*img_in+IFFT(freq_in))))
          freq_conv(float): nonlinear(conv(BN(alpha*freq_in+FFT(img_in))))

        """
        if(self.hyp_conv):
            img_in, freq_in, hyp_tensor = x
        else:
            img_in, freq_in = x

        img_in_as_freq = utils.convert_channels_to_frequency_domain(img_in)
        freq_in_as_img = utils.convert_channels_to_image_domain(freq_in)

        img_feat = self.img_mix([img_in, freq_in_as_img])
        k_feat = self.freq_mix([freq_in, img_in_as_freq])

        for i in range(self.num_convs):
            if(self.shift):
                img_feat = tf.signal.ifftshift(img_feat, axes=(1,2))
            if(self.hyp_conv):
                img_conv = self.img_bnconvs[i]((img_feat, hyp_tensor))
            else:
                img_conv = self.img_bnconvs[i](img_feat)
            img_feat = get_nonlinear_layer('relu')(img_conv)

            if(self.shift):
                img_feat = tf.signal.fftshift(img_feat, axes=(1,2))

            if(self.hyp_conv):
                k_conv = self.freq_bnconvs[i]((k_feat, hyp_tensor))
            else:
                k_conv = self.freq_bnconvs[i](k_feat)
            k_feat = get_nonlinear_layer('3-piece')(k_conv)

        return (img_feat, k_feat)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][:3] + [self.features],
                input_shape[1][:3] + [self.features])
