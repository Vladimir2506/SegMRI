import tensorflow as tf
import numpy as np
import os
import glob

class Dataset_2d(object):

    def __init__(self, record_path, batch_size, shuffle_buffer):
        
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.record_path = record_path
        self.train_records_path = os.path.join(self.record_path, 'train')
        self.test_records_path = os.path.join(self.record_path, 'test')
        self.train_record_names = sorted(glob.glob(os.path.join(self.train_records_path, '*.tfrecord')))
        self.test_record_names = sorted(glob.glob(os.path.join(self.test_records_path, '*.tfrecord')))

    @tf.function
    def _decode_and_reshape_affine(self, name, shape, dtype):

        name = tf.io.decode_raw(name, dtype)
        name = tf.reshape(name, [shape[0], shape[1]])
        # name = tf.cast(name, tf.float32)

        return name

    @tf.function
    def _decode_and_reshape(self, name, shape, dtype):

        name = tf.io.decode_raw(name, dtype)
        name = tf.reshape(name, [shape[0], shape[1], 1])
        name = tf.cast(name, tf.float32)

        return name
    
    @tf.function
    def _parse(self, example_proto):
        
        features = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'image_shape': tf.io.FixedLenFeature([2], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string), 
            'label_shape': tf.io.FixedLenFeature([2], tf.int64)
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)
        
        label = self._decode_and_reshape(parsed_features['label'], parsed_features['label_shape'], tf.uint8)
        image = self._decode_and_reshape(parsed_features['image'], parsed_features['image_shape'], tf.int16)
        
        label = tf.cast(label, tf.int32)
        image = tf.cast(image, tf.float32)

        return image, label
    
    @tf.function
    def _decode_and_reshape_3d(self, name, shape, dtype):

        name = tf.io.decode_raw(name, dtype)
        name = tf.reshape(name, [shape[0], shape[1], shape[2], 1])
        name = tf.cast(name, tf.float32)

        return name
    
    @tf.function
    def _parse_3d(self, example_proto):
        
        features = {
            'affine': tf.io.FixedLenFeature([], tf.string),
            'affine_shape': tf.io.FixedLenFeature([2], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string), 
            'image_shape': tf.io.FixedLenFeature([3], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string), 
            'label_shape': tf.io.FixedLenFeature([3], tf.int64)
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)
        
        label = self._decode_and_reshape_3d(parsed_features['label'], parsed_features['label_shape'], tf.uint8)
        image = self._decode_and_reshape_3d(parsed_features['image'], parsed_features['image_shape'], tf.int16)
        affine = self._decode_and_reshape_affine(parsed_features['affine'], parsed_features['affine_shape'], tf.float32)

        label = tf.cast(label, tf.int32)
        image = tf.cast(image, tf.float32)

        return image, label, affine
    

    def get_loader(self, training):

        if training:
            return tf.data.TFRecordDataset(self.train_record_names).repeat().shuffle(self.shuffle_buffer).map(self._parse).prefetch(2 * self.batch_size).batch(self.batch_size)
        else:
            # return tf.data.TFRecordDataset(self.test_record_names).repeat().shuffle(self.shuffle_buffer).map(self._parse).prefetch(2 * self.batch_size).batch(self.batch_size)
            return tf.data.TFRecordDataset(self.test_record_names).map(self._parse_3d).prefetch(2).batch(1)