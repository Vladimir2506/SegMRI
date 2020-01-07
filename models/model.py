import tensorflow as tf
import pdb


class ConvBNReLU(tf.keras.Model):

    def __init__(self, channels, regularizer, initializer):
        
        super(ConvBNReLU, self).__init__()

        self.conv = tf.keras.layers.Conv3D(channels, 3, 1, 'SAME', kernel_regularizer=regularizer, kernel_initializer=initializer)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(negative_slope=0.1)
    
    @tf.function
    def call(self, x, training):

        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class Conv_block(tf.keras.Model):
    """
    convolution block
    """

    def __init__(self, channels, regularizer, initializer):

        super(Conv_block, self).__init__()
        self.channels = channels

        self.conv1 = tf.keras.layers.Convolution3D(
            channels, kernel_size=3, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)

        self.conv2 = tf.keras.layers.Convolution3D(
            channels, kernel_size=3, strides=1, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu2(out)

        return out

class up_conv(tf.keras.Model):
    """
    Up Convolution Block
    """

    def __init__(self, channels, regularizer, initializer):

        super(up_conv, self).__init__()
        self.channels = channels

        self.upSample = tf.keras.layers.Conv3DTranspose(
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

class U_Net(tf.keras.Model):
    
    def __init__(self, num_classes, basic_channels, weight_decay, dropout_rate):

        super(U_Net, self).__init__()

        self.basic_channels = basic_channels
        self.regularizer = tf.keras.regularizers.l2(weight_decay)
        self.num_classes = num_classes
        self.initializer = tf.keras.initializers.he_normal()
        self.dropout_rate = dropout_rate

        self.Maxpool1 = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)
        self.Maxpool2 = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)
        self.Maxpool3 = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)
        self.Maxpool4 = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)
        
        self.Conv1 = Conv_block(self.basic_channels, self.regularizer, self.initializer)
        self.Conv2 = Conv_block(self.basic_channels*2, self.regularizer, self.initializer)
        self.Conv3 = Conv_block(self.basic_channels*4, self.regularizer, self.initializer)
        self.Conv4 = Conv_block(self.basic_channels*8, self.regularizer, self.initializer)
        self.Conv5 = Conv_block(self.basic_channels*16, self.regularizer, self.initializer)
        
        self.Up5 = up_conv(self.basic_channels*8, self.regularizer, self.initializer)
        self.Up_Conv5 = Conv_block(self.basic_channels*8, self.regularizer, self.initializer)
        
        self.Up4 = up_conv(self.basic_channels*4, self.regularizer, self.initializer)
        self.Up_Conv4 = Conv_block(self.basic_channels*4, self.regularizer, self.initializer)
        
        self.Up3 = up_conv(self.basic_channels*2, self.regularizer, self.initializer)
        self.Up_Conv3 = Conv_block(self.basic_channels*2, self.regularizer, self.initializer)

        self.Up2 = up_conv(self.basic_channels, self.regularizer, self.initializer)
        self.Up_Conv2 = Conv_block(self.basic_channels, self.regularizer, self.initializer)
        
        self.Conv = tf.keras.layers.Conv3D(num_classes, kernel_size=1, strides=1, padding='SAME', kernel_regularizer=self.regularizer, kernel_initializer=self.initializer)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    
    @tf.function
    def call(self, x, training):
        
        e1 = self.Conv1(x, training=training)
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, training=training)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, training=training)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4, training=training)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5, training=training)
        
        d5 = self.Up5(e5, training=training)
        d5 = tf.concat([e4, d5], axis=4)
        d5 = self.Up_Conv5(d5, training=training)
        
        d4 = self.Up4(d5, training=training)
        d4 = tf.concat([e3, d4], axis=4)
        d4 = self.Up_Conv4(d4, training=training)
        
        d3 = self.Up3(d4, training=training)
        d3 = tf.concat([e2, d3], axis=4)
        d3 = self.Up_Conv3(d3, training=training)
        
        d2 = self.Up2(d3, training=training)
        d2 = tf.concat([e1, d2], axis=4)
        d2 = self.Up_Conv2(d2, training=training)

        d2 = self.dropout(d2, training=training)
        
        out = self.Conv(d2, training=training)

        return out

class SimpleUNet(tf.keras.Model):

    def __init__(self, num_classes, basic_channels, weight_decay):
        
        super(SimpleUNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = basic_channels
        self.weight_decay = weight_decay
        self.initializer = tf.keras.initializers.he_normal()
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.conv_in = tf.keras.layers.Conv3D(self.num_features, 5, 1, 'same', activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv1 = ConvBNReLU(self.num_features, self.regularizer, self.initializer)
        self.max_pool1 = tf.keras.layers.MaxPool3D(2, 2, 'same')
        self.conv2 = ConvBNReLU(self.num_features * 2, self.regularizer, self.initializer)
        self.max_pool2 = tf.keras.layers.MaxPool3D(2, 2, 'same')
        self.conv3 = ConvBNReLU(self.num_features * 4, self.regularizer, self.initializer)
        self.max_pool3 = tf.keras.layers.MaxPool3D(2, 2, 'same')
        self.conv4 = ConvBNReLU(self.num_features * 8, self.regularizer, self.initializer)
        self.upsample_3 = tf.keras.layers.UpSampling3D(2)
        self.conv5 = ConvBNReLU(self.num_features * 4, self.regularizer, self.initializer)
        self.upsample_2 = tf.keras.layers.UpSampling3D(2)
        self.conv6 = ConvBNReLU(self.num_features * 2, self.regularizer, self.initializer)
        self.upsample_1 = tf.keras.layers.UpSampling3D(2)
        self.conv7 = ConvBNReLU(self.num_features, self.regularizer, self.initializer)
        self.conv_out = tf.keras.layers.Conv3D(self.num_classes, 1, 1, 'same', activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    @tf.function
    def call(self, x, training):

        x = self.conv_in(x)
        x = self.conv1(x, training=training)
        x1 = self.max_pool1(x)
        x1 = self.conv2(x1, training=training)
        x2 = self.max_pool2(x1)
        x2 = self.conv3(x2, training=training)
        x3 = self.max_pool3(x2)
        x3 = self.conv4(x3, training=training)
        x3 = self.upsample_3(x3)
        x2 = tf.concat([x2, x3], axis=4)
        x2 = self.conv5(x2, training=training)
        x2 = self.upsample_2(x2)
        x1 = tf.concat([x1, x2], axis=4)
        x1 = self.conv6(x1, training=training)
        x1 = self.upsample_1(x1)
        x = tf.concat([x, x1], axis=4)
        x = self.conv7(x, training=training)
        x = self.conv_out(x)

        return x

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

class U_Net_2d(tf.keras.Model):
    
    def __init__(self, num_classes, basic_channels, weight_decay, dropout_rate):

        super(U_Net_2d, self).__init__()

        self.basic_channels = basic_channels
        self.regularizer = tf.keras.regularizers.l2(weight_decay)
        self.num_classes = num_classes
        self.initializer = tf.keras.initializers.he_normal()
        self.dropout_rate = dropout_rate

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
        self.Up_Conv4 = Conv_block_2d(self.basic_channels*4, self.regularizer, self.initializer)
        
        self.Up3 = up_conv_2d(self.basic_channels*2, self.regularizer, self.initializer)
        self.Up_Conv3 = Conv_block_2d(self.basic_channels*2, self.regularizer, self.initializer)

        self.Up2 = up_conv_2d(self.basic_channels, self.regularizer, self.initializer)
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
        d4 = tf.concat([e3, d4], axis=3)
        d4 = self.Up_Conv4(d4, training=training)
        
        d3 = self.Up3(d4, training=training)
        d3 = tf.concat([e2, d3], axis=3)
        d3 = self.Up_Conv3(d3, training=training)
        
        d2 = self.Up2(d3, training=training)
        d2 = tf.concat([e1, d2], axis=3)
        d2 = self.Up_Conv2(d2, training=training)

        d2 = self.dropout(d2, training=training)
        
        out = self.Conv(d2, training=training)

        return out
