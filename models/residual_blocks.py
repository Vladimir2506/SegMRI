import tensorflow as tf

class Residual2Dv1(tf.keras.Model):

    def __init__(self, channels, regularizer, initializer):

        super(Residual2Dv1, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(channels, 3, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        x += res
        x = self.relu2(x)

        return x

class Residual2Dv2(tf.keras.Model):

    def __init__(self, channels, regularizer, initializer):

        super(Residual2Dv2, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        res = self.bn1(x)
        res = self.relu1(res)
        res = self.conv1(res)
        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv2(res)

        return x + res

class Residual3Dv1(tf.keras.Model):

    def __init__(self, channels, regularizer, initializer, kd=None, ksize=3):

        super(Residual3Dv1, self).__init__()
        
        ksize = [ksize, ksize, kd] if kd else ksize

        self.conv1 = tf.keras.layers.Conv3D(channels, ksize, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)
        self.conv2 = tf.keras.layers.Conv3D(channels, ksize, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        x += res
        x = self.relu2(x)

        return x

class Residual3Dv2(tf.keras.Model):

    def __init__(self, channels, regularizer, initializer, kd=None, ksize=3):

        super(Residual3Dv2, self).__init__()

        ksize = [ksize, ksize, kd] if kd else ksize
        
        self.conv1 = tf.keras.layers.Conv3D(channels, ksize, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU(negative_slope=0.1)
        self.conv2 = tf.keras.layers.Conv3D(channels, ksize, 1, 'same', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU(negative_slope=0.1)

    def call(self, x, training):

        res = self.bn1(x)
        res = self.relu1(res)
        res = self.conv1(res)
        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv2(res)

        return x + res