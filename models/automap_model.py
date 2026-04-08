"""
AUTOMAP: Automated Transform by Manifold Approximation

Based on:
    Zhu B, Liu JZ, Cauley SF, Rosen BR, Rosen MS. "Image reconstruction by
    domain-transform manifold learning."
    Nature 555, 487–492 (2018). https://doi.org/10.1038/nature25988

Architecture: two FC layers perform a learned domain transform from flattened
k-space to image space, followed by a shallow Conv network for denoising.

In this work two instances of AUTOMAP_Basic_Model4 are trained separately for
the real and imaginary channels, then combined into a complex image.

Input: 1D vector of concatenated real and imaginary k-space samples
       [batch, 2 * n_samples].
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def AUTOMAP_Basic_Model4(config):
    """
    FC(tanh) → FC → reshape → ZeroPad → Conv(tanh) → Conv(relu) → Conv(relu,skip) → ConvTranspose

    config fields: fc_input_dim, fc_hidden_dim, fc_output_dim (= im_h * im_w), im_h, im_w.
    """
    fc_1 = keras.Input(shape=(config.fc_input_dim,), name='input')

    with tf.device('/gpu:0'):
        fc_2 = layers.Dense(config.fc_hidden_dim, activation='tanh', use_bias=True)(fc_1)
    with tf.device('/gpu:1'):
        fc_3 = layers.Dense(config.fc_output_dim, use_bias=True)(fc_2)

    fc_3 = layers.Reshape((config.im_h, config.im_w, 1))(fc_3)
    fc_3 = layers.ZeroPadding2D(4)(fc_3)  # pad by 4 on each side → (im_h+8, im_w+8)

    c_1 = layers.Conv2D(128, 5, strides=1, padding='same', activation='tanh')(fc_3)
    c_2 = layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(c_1)
    c_3 = layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(c_2)

    c_32 = layers.Add()([c_3, c_1])  # skip connection

    c_4 = layers.Conv2DTranspose(1, 5, strides=1, padding='same', use_bias=True)(c_32)

    output = layers.Reshape(((config.im_h + 8) * (config.im_w + 8),))(c_4)

    return keras.Model(inputs=fc_1, outputs=[c_3, output], name='automap_basic_4')
