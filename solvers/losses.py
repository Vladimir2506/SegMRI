import tensorflow as tf

import pdb

class BalancedLoss(object):

    def __init__(self, name, bce_weight):
        self.bce_weight = bce_weight
        self.name = name

    def __call__(self, y_true, y_logit):
        
        y_pred = tf.nn.softmax(y_logit)[...,0:1]
        y_true = tf.reshape(y_true, shape=[-1])
        y_pred = tf.reshape(y_pred, shape=[-1])
        
        if self.bce_weight == 0:
            return dice_loss(y_true, y_pred)
        elif self.bce_weight == 1:
            return bce_loss(y_true, y_pred)
        else:
            return self.bce_weight * bce_loss(y_true, y_pred) + (1 - self.bce_weight) * dice_loss(y_true, y_pred)
    
@tf.function
def dice_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    pos_num = tf.reduce_sum(y_true * y_pred) + 1.0
    pos_den = tf.reduce_sum(y_true ** 2) + tf.reduce_sum(y_pred ** 2) + 1.0

    # neg_num = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred)) + 1.0
    # neg_den = tf.reduce_sum((1.0 - y_true) ** 2) + tf.reduce_sum((1.0 - y_pred) ** 2) + 1.0

    return 1.0 - 2.0 * pos_num / pos_den

@tf.function
def bce_loss(y_true, y_pred):
    
    N_all = tf.size(y_true)
    
    pos_index = tf.where(y_true == 1)
    neg_index = tf.where(y_true == 0)

    N_pos = tf.size(pos_index)
    N_neg = N_all - N_pos
    # print(f'pos:{N_pos.numpy()}, neg:{N_neg.numpy()}')
    w_pos = tf.cast(N_neg, tf.float32) / tf.cast(N_all, tf.float32)
    w_neg = tf.cast(N_pos, tf.float32) / tf.cast(N_all, tf.float32)

    raw_loss = tf.losses.binary_crossentropy(y_true, y_pred)
    mask = tf.scatter_nd(pos_index, tf.ones([N_pos]) * w_pos, shape=y_pred.shape)
    mask = tf.tensor_scatter_nd_update(mask, neg_index, tf.ones([N_neg]) * w_neg)
    mask /= tf.cast(N_all, tf.float32)

    # return tf.reduce_mean(raw_loss)
    return tf.reduce_sum(raw_loss * mask)

class LRDEHL(object):

    def __init__(self, name, neg_rate, hard_rate, loss_order, min_neg_num):

        self.name = name
        self.ideal_neg_rate = neg_rate
        self.ideal_pos_rate = 1.0 - self.ideal_neg_rate
        self.hard_rate = hard_rate
        self.loss_order = loss_order
        self.min_neg_num = min_neg_num

    def __call__(self, y_true, y_logit):

        y_pred = tf.nn.softmax(y_logit)[...,0:1]
        return self.forward(y_true, y_pred)

    @tf.function
    def forward(self, y_true, y_pred):
        
        # Label Related Dropout
        y_true = tf.cast(y_true, tf.uint8)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        pos_index = tf.where(y_true == 1)
        neg_index = tf.where(y_true == 0)

        nb_pos_samples = tf.size(pos_index)
        nb_neg_to_sample = tf.cast(nb_pos_samples, tf.float32) / self.ideal_pos_rate * self.ideal_neg_rate
        nb_neg_to_sample = tf.maximum(nb_neg_to_sample, tf.cast(self.min_neg_num, tf.float32))

        nb_hard_neg_to_sample = tf.cast(nb_neg_to_sample * self.hard_rate, tf.int32)
        nb_easy_neg_to_sample = tf.cast(nb_neg_to_sample, tf.int32) - nb_hard_neg_to_sample
        
        neg_preds = tf.gather(y_pred, neg_index)

        neg_preds = tf.reshape(neg_preds, [1, -1])
        _, hard_index = tf.nn.top_k(neg_preds, nb_hard_neg_to_sample)
        hard_index = tf.reshape(hard_index, [-1])

        sampled_hard_neg_index = tf.gather(neg_index, hard_index)
        sample_easy_neg_index = tf.gather(neg_index, tf.random.uniform([nb_easy_neg_to_sample], minval=0, maxval=tf.size(neg_index), dtype=tf.int32))
        sampled_index = tf.concat([pos_index ,sampled_hard_neg_index, sample_easy_neg_index], axis=0)
        sampled_index, _ = tf.unique(tf.squeeze(sampled_index), tf.int64)
        sampled_index = tf.reshape(sampled_index, [-1, 1])
        mask = tf.scatter_nd(sampled_index, tf.ones_like(sampled_index, dtype=tf.bool), [tf.size(y_pred), 1])
        mask = tf.squeeze(mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.boolean_mask(y_true, mask)
        
        # Element Hinge Loss
        if self.loss_order == 1:
            loss_val = tf.losses.mean_absolute_error(y_true, y_pred)
        elif self.loss_order == 2:
            loss_val = tf.losses.mean_squared_error(y_true, y_pred)
        else:
            raise RuntimeError('Hinge loss order must be 1 or 2.')

        # loss_val /= tf.maximum(nb_pos_samples, nb_neg_to_sample)

        return loss_val

def get_loss(cfg_loss):

    name = cfg_loss['name']
    kwargs = cfg_loss['kwargs']
    name_ =  name.lower()

    if name_ == 'lrdehl':
        return LRDEHL(name=name, **kwargs)
    elif name_ == 'balanced':
        return BalancedLoss(name=name, **kwargs)
    else:
        raise NotImplementedError(f'Loss {name} is not supported.')