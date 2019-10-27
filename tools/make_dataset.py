import os
import nibabel
import tensorflow as tf
import numpy as np
import argparse
import pdb

'''
    We suppose the raw datasets have structures like this.
    raw_root
    ├─FZMC00x
    │       .DS_Store
    │       artery.nii
    │       delay.nii
    │       label.nii
    │       portal.nii
    │       pre.nii
    │
    └─FZMC00y
            .DS_Store
            artery.nii
            delay.nii
            label.nii
            portal.nii
            pre.nii
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_root',
        type=str,
        help='The path to raw dataset with *.nii format.'
    )
    parser.add_argument(
        '--record_root',
        type=str,
        help='The path to generated tfrecords.'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        help='The max samples in one record.'
    )
    args = parser.parse_args()
    
    raw_root = args.raw_data_root
    record_root = args.record_root
    nb_samples = args.max_samples

    if not os.path.exists(record_root):
        os.makedirs(record_root)

    data_names = sorted(os.listdir(raw_root))

    fid = 1
    writer = tf.io.TFRecordWriter(os.path.join(record_root, f'data_{fid:02d}.tfrecord'))

    for i, data_name in enumerate(data_names):

        data_name_root = os.path.join(raw_root, data_name)

        example = make_nii_example(data_name_root)

        writer.write(example.SerializeToString())

        print(f'{data_name:s} process done.')

        if (i + 1) % nb_samples == 0:
            writer.close()
            fid += 1
            if i < len(data_names) - 1: 
                writer = tf.io.TFRecordWriter(os.path.join(record_root, f'data_{fid:02d}.tfrecord'))
        
        if (i + 1) == len(data_names):
            writer.close()


def make_nii_example(data_name_root):

    artery_nii = nibabel.load(os.path.join(data_name_root, 'artery.nii'))
    delay_nii = nibabel.load(os.path.join(data_name_root, 'delay.nii'))
    portal_nii = nibabel.load(os.path.join(data_name_root, 'portal.nii'))
    pre_nii = nibabel.load(os.path.join(data_name_root, 'pre.nii'))
    label_nii = nibabel.load(os.path.join(data_name_root, 'label.nii'))

    artery = np.asarray(artery_nii.get_data())
    delay = np.asarray(delay_nii.get_data())
    portal = np.asarray(portal_nii.get_data())
    pre = np.asarray(pre_nii.get_data())
    label = np.asarray(label_nii.get_data())

    return tf.train.Example(
    	features=tf.train.Features(
        	feature={
            	'artery': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(artery)])),
                'artery_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(artery.shape))),
            	'delay': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(delay)])),
                'delay_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(delay.shape))),
            	'portal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(portal)])),
                'portal_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(portal.shape))),
            	'pre': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(pre)])),
                'pre_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(pre.shape))),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(label)])),
                'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(label.shape)))
        	}
    	)
	)

if __name__ == "__main__":
    main()
