import tensorflow as tf

class DiceMeter(tf.metrics.Metric):

    def __init__(self):
        super(DiceMeter, self).__init__()
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')
        
    def update_state(self, y_true, y_pred, threshold=0.5):
        
        N_samples = tf.cast(y_true.shape[0], tf.float32)
        self.dice.assign_add(dice_coef(y_true, y_pred, threshold) * N_samples)
        self.samples.assign_add(N_samples)
    
    def result(self):
        return self.dice / self.samples

class BinaryPRMeter(tf.metrics.Metric):

    def __init__(self):
        super(BinaryPRMeter, self).__init__()
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def update_state(self, y_true, y_pred, threshold=0.5):
        
        N_samples = tf.cast(y_true.shape[0], tf.float32)
        precision, recall = precision_recall(y_true, y_pred, threshold)
        self.precision.assign_add(precision * N_samples)
        self.recall.assign_add(recall * N_samples)
        self.samples.assign_add(N_samples)
    
    def result(self):
        return self.precision / self.samples, self.recall / self.samples

@tf.function
def dice_coef(y_true, y_pred, threshold=0.5):

    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    numerator = 2.0 * tf.reduce_sum(y_pred * y_true) + 1e-8
    denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + 1e-8

    coef = numerator / denominator
    
    return coef

@tf.function
def precision_recall(y_true, y_pred, threshold=0.5):

    y_pred = tf.cast(y_pred > threshold, tf.bool)
    y_true = tf.cast(y_true, tf.bool)

    tp = tf.math.count_nonzero(y_pred & y_true, dtype=tf.float32)
    fp = tf.math.count_nonzero(y_pred & ~y_true, dtype=tf.float32)
    fn = tf.math.count_nonzero(~y_pred & y_true, dtype=tf.float32)
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)

    return precision, recall
    