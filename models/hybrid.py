import tensorflow as tf
import pdb

from .residual_blocks import Residual3Dv2, Residual3Dv1

class UpConv3DHybrid(tf.keras.Model):

    def __init__(self, rate, channels, kd, initializer, regularizer):

        super(UpConv3DHybrid, self).__init__()

        self.rate = rate
        self.up = tf.keras.layers.UpSampling2D(self.rate, interpolation='bilinear')
        self.conv = tf.keras.layers.Conv3D(channels, (1, 1, kd), 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
    
    def call(self, x, y):

        N, H, W, D, C = x.shape
        x = tf.reshape(x, [N, H, W, -1])
        x = self.up(x)
        x = tf.reshape(x, [N, H * self.rate, W * self.rate, D, C])
        out = tf.concat([y, x], axis=4)
        out = self.conv(out)
        return out 

class HybridUNetX(tf.keras.Model):

    def __init__(self, basic_channels, k_depth, blocks, num_classes, weight_decay):

        super(HybridUNetX, self).__init__()

        self.basic_channels = basic_channels
        self.num_classes = num_classes
        self.regularizer = tf.keras.regularizers.l2(weight_decay)
        self.initializer = tf.keras.initializers.he_normal()
        self.kd = k_depth
        self.blocks = blocks

        self.ksize_in = [5, 5, self.kd]
        self.ksize = [3, 3, self.kd]
        self.ksize_out = [1, 1, self.kd]

        self.conv_in = tf.keras.layers.Conv3D(self.basic_channels, self.ksize_in, 1, 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.stage1_1 = Residual3Dv2(self.basic_channels, self.regularizer, self.initializer, self.kd)
        self.stage1_2 = Residual3Dv2(self.basic_channels, self.regularizer, self.initializer, self.kd)

        self.ds_1 = tf.keras.layers.Conv3D(self.basic_channels * 2, self.ksize, (2, 2, 1), 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us_1 = UpConv3DHybrid(2, self.basic_channels, self.kd, self.initializer, self.regularizer)

        self.stage2_1 = Residual3Dv2(self.basic_channels * 2, self.regularizer, self.initializer, self.kd)
        self.stage2_2 = Residual3Dv2(self.basic_channels * 2, self.regularizer, self.initializer, self.kd)

        self.ds_2 = tf.keras.layers.Conv3D(self.basic_channels * 4, self.ksize, (2, 2, 1), 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us_2 = UpConv3DHybrid(2, self.basic_channels * 2, self.kd, self.initializer, self.regularizer)

        self.stage3_1 = Residual3Dv2(self.basic_channels * 4, self.regularizer, self.initializer, self.kd)
        self.stage3_2 = Residual3Dv2(self.basic_channels * 4, self.regularizer, self.initializer, self.kd)

        self.ds_3 = tf.keras.layers.Conv3D(self.basic_channels * 4, self.ksize, (2, 2, 1), 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us_3 = UpConv3DHybrid(2, self.basic_channels * 4, self.kd, self.initializer, self.regularizer)

        self.stage4 = tf.keras.Sequential([
            Residual3Dv2(self.basic_channels * 4, self.regularizer, self.initializer, self.kd) for _ in range(self.blocks)
        ])
        # pdb.set_trace()
        self.conv_out = tf.keras.layers.Conv3D(self.num_classes, self.ksize_out, 1, 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, x, training):

        x1 = self.conv_in(x)
        x1 = self.stage1_1(x1, training=training)
        x2 = self.ds_1(x1)
        x2 = self.stage2_1(x2, training=training)
        x3 = self.ds_2(x2)
        x3 = self.stage3_1(x3, training=training)
        x4 = self.ds_3(x3)
        x4 = self.stage4(x4, training=training)
        x3 = self.us_3(x4, x3)
        x3 = self.stage3_2(x3, training=training)
        x2 = self.us_2(x3, x2)
        x2 = self.stage2_2(x2, training=training)
        x1 = self.us_1(x2, x1)
        x1 = self.stage1_2(x1, training=training)
        x = self.conv_out(x1)

        return x


