import tensorflow as tf
import pdb

def decode_and_reshape(name, shape):

    name = tf.io.decode_raw(name, tf.int16)
    name = tf.reshape(name, [shape[0], shape[1], shape[2]])
    name = tf.cast(name, tf.float32)

    return name

def parse_function(example_proto):

    features = {
        'artery': tf.io.FixedLenFeature([], tf.string), 
        'artery_shape': tf.io.FixedLenFeature([3], tf.int64),
        'portal': tf.io.FixedLenFeature([], tf.string), 
        'portal_shape': tf.io.FixedLenFeature([3], tf.int64),
        'pre': tf.io.FixedLenFeature([], tf.string), 
        'pre_shape': tf.io.FixedLenFeature([3], tf.int64),
        'delay': tf.io.FixedLenFeature([], tf.string), 
        'delay_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string), 
        'label_shape': tf.io.FixedLenFeature([3], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(example_proto, features)
    
    label = decode_and_reshape(parsed_features['label'], parsed_features['label_shape'])
    artery = decode_and_reshape(parsed_features['artery'], parsed_features['artery_shape'])
    pre = decode_and_reshape(parsed_features['pre'], parsed_features['pre_shape'])
    delay = decode_and_reshape(parsed_features['delay'], parsed_features['delay_shape'])
    portal = decode_and_reshape(parsed_features['portal'], parsed_features['portal_shape'])
    
    return artery, delay, portal, pre, label

reader = tf.data.TFRecordDataset(['dataset/record/data_02.tfrecord'])
reader = reader.map(parse_function).batch(1)

for exam in reader:
    pdb.set_trace()
