import tensorflow as tf
import pdb

class up_conv_2d(tf.keras.Model):
    """
    Up Convolution Block
    """

    def __init__(self, channels, regularizer, initializer):

        super(up_conv_2d, self).__init__()
        self.channels = channels

        self.upSample = tf.keras.layers.Conv2DTranspose(
            channels, kernel_size=3, strides=2, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        out = self.upSample(x)
        # pdb.set_trace()
        out = self.bn(out, training=training)
        out = self.relu(out)

        return out

class Conv_block_2d(tf.keras.Model):
    """
    convolution block
    """

    def __init__(self, channels, regularizer, initializer):

        super(Conv_block_2d, self).__init__()
        self.channels = channels

        self.conv1 = tf.keras.layers.Convolution2D(
            channels, kernel_size=3, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)

        # self.conv2 = tf.keras.layers.Convolution2D(
        #    channels, kernel_size=3, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer
        # )
        # self.bn2 = tf.keras.layers.BatchNormalization()
        # self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        
        # out = self.conv2(out)
        # out = self.bn2(out, training=training)
        # out = self.relu2(out)

        return out


class attention_block_2d(tf.keras.Model):
    """
    Attention block
    """

    def __init__(self, channels, regularizer, initializer):

        super(attention_block_2d, self).__init__()

        self.channels = channels

        self.conv1 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer)
        self.conv2 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer)
        self.conv3 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, g, x, training):

        g1 = self.conv1(g)
        g1 = self.bn1(g1)

        w1 = self.conv2(x)
        w1 = self.bn2(w1)

        psi = tf.keras.layers.ReLU()

        psi = self.conv3(psi)
        psi = self.bn3(psi)
        psi = tf.nn.sigmoid(psi)

        out = x + psi

        return out

class Atten_Unet_2d(tf.keras.Model):
    # Attention Unet

    def __init__(self, num_classes, basic_channels, weight_decay):

        super(Atten_Unet_2d, self).__init__()

        self.basic_channels = basic_channels
        self.regularizer = tf.keras.regularizers.l2(weight_decay)
        self.num_classes = num_classes
        self.initializer = tf.keras.initializers.he_normal()
        # self.dropout_rate = dropout_rate

        self.Maxpool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.Maxpool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.Maxpool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # self.Maxpool4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        
        self.Conv1 = Conv_block_2d(self.basic_channels, self.regularizer, self.initializer)
        self.Conv2 = Conv_block_2d(self.basic_channels*2, self.regularizer, self.initializer)
        self.Conv3 = Conv_block_2d(self.basic_channels*4, self.regularizer, self.initializer)
        self.Conv4 = Conv_block_2d(self.basic_channels*8, self.regularizer, self.initializer)
        # self.Conv5 = Conv_block_2d(self.basic_channels*16, self.regularizer, self.initializer)
        
        # self.Up5 = up_conv_2d(self.basic_channels*8, self.regularizer, self.initializer)
        # self.Up_Conv5 = Conv_block_2d(self.basic_channels*8, self.regularizer, self.initializer)
        
        self.Up4 = up_conv_2d(self.basic_channels*4, self.regularizer, self.initializer)
        self.Att4 = attention_block_2d(self.basic_channels*4, self.regularizer, self.initializer)
        self.Up_Conv4 = Conv_block_2d(self.basic_channels*4, self.regularizer, self.initializer)
        
        self.Up3 = up_conv_2d(self.basic_channels*2, self.regularizer, self.initializer)
        self.Att3 = attention_block_2d(self.basic_channels*2, self.regularizer, self.initializer)
        self.Up_Conv3 = Conv_block_2d(self.basic_channels*2, self.regularizer, self.initializer)

        self.Up2 = up_conv_2d(self.basic_channels, self.regularizer, self.initializer)
        self.Att2 = attention_block_2d(self.basic_channels, self.regularizer, self.initializer)
        self.Up_Conv2 = Conv_block_2d(self.basic_channels, self.regularizer, self.initializer)
        
        self.Conv = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding='SAME', kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    
    # @tf.function
    def call(self, x, training):
        
        e1 = self.Conv1(x, training=training)
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, training=training)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, training=training)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4, training=training)
        
        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5, training=training)
        
        # d5 = self.Up5(e5, training=training)
        # d5 = tf.concat([e4, d5], axis=3)
        # d5 = self.Up_Conv5(d5, training=training)
        
        d4 = self.Up4(e4, training=training)
        x3 = self.Att4(g=d4, x=e3)
        d4 = tf.concat([x3, d4], axis=3)
        d4 = self.Up_Conv4(d4, training=training)
        
        d3 = self.Up3(d4, training=training)
        x2 = self.Att3(g=d3, x=e2)
        d3 = tf.concat([x2, d3], axis=3)
        d3 = self.Up_Conv3(d3, training=training)
        
        d2 = self.Up2(d3, training=training)
        x1 = self.Att2(g=d2, x=e1)
        d2 = tf.concat([x1, d2], axis=3)
        d2 = self.Up_Conv2(d2, training=training)

        d2 = self.dropout(d2, training=training)
        
        out = self.Conv(d2, training=training)

        return out

