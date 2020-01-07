import os
import nibabel
import tensorflow as tf
import numpy as np
import argparse
import glob

import pdb

from skimage.transform import resize

'''
    We suppose the raw datasets have structures like this.
    raw_root
    └─FZMC00x
            .DS_Store
            artery.nii
            delay.nii
            label.nii
            portal.nii
            pre.nii
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', type=str, help='The path to raw dataset with *.nii format.')
    parser.add_argument('--record_root', type=str, help='The path to generated tfrecords.')
    parser.add_argument('--nb_samples', type=int, help='Number of total generated samples.')
    parser.add_argument('--resample_size', type=int, nargs='+', help='Size of volumetric data to resample, need 3 args in H, W, D.')
    parser.add_argument('--patch_size', type=int, help='Size of generated patches.')
    parser.add_argument('--test_rate', type=float, help='The split rate for test data.')
    parser.add_argument('--samples_per_record', type=int, help='Number of patches in one tfrecord.')
    parser.add_argument('--phase', type=str, choices=['artery', 'pre', 'delay', 'portal'], help='Specify one phase of data to generate.')
    args = parser.parse_args()

    # 0. Paths preparation
    record_root = os.path.join(args.record_root, args.phase)
    train_root = os.path.join(record_root, 'train')
    test_root = os.path.join(record_root, 'test')

    if not os.path.exists(record_root):
        os.makedirs(record_root)
        
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    
    if not os.path.exists(test_root):
        os.makedirs(test_root)

    vol_names = sorted(glob.glob(os.path.join(args.raw_data_root, '*', f'{args.phase}.nii')))
    seg_names = sorted(glob.glob(os.path.join(args.raw_data_root, '*', 'label.nii')))

    assert len(vol_names) == len(seg_names)
    nb_raw_data = len(vol_names)
    nb_test_data = int(nb_raw_data * args.test_rate)
    nb_train_data = nb_raw_data - nb_test_data
    
    train_vol_names = vol_names[0:nb_train_data]
    train_seg_names = seg_names[0:nb_train_data]
    test_vol_names = vol_names[nb_train_data:]
    test_seg_names = seg_names[nb_train_data:]

    with open(os.path.join(record_root, 'train_split.txt'), 'w') as f:
        for train_vol_name, train_seg_name in zip(train_vol_names, train_seg_names):
            f.write(f'{train_vol_name} {train_seg_name}\n')

    with open(os.path.join(record_root, 'test_split.txt'), 'w') as f:
        for test_vol_name, test_seg_name in zip(test_vol_names, test_seg_names):
            f.write(f'{test_vol_name} {test_seg_name}\n')

    # 1. Test data generation
    for test_data_idx, (test_vol_name, test_seg_name) in enumerate(zip(test_vol_names, test_seg_names)):
        test_example = make_test_example(test_vol_name, test_seg_name, args.resample_size)
        writer = tf.io.TFRecordWriter(os.path.join(test_root, f'data_{test_data_idx}.tfrecord'))
        writer.write(test_example.SerializeToString())
        writer.close()
        print(f'[{test_data_idx + 1} / {nb_test_data}] Test data generated.')
    # pdb.set_trace()
    # 2. Train data generation
    samples_per_volume = np.math.ceil(args.nb_samples / nb_train_data)
    actual_sampled = 0
    for train_data_idx, (train_vol_name, train_seg_name) in enumerate(zip(train_vol_names, train_seg_names)):
        train_examples = make_train_examples(train_vol_name, train_seg_name, args.resample_size, args.patch_size, samples_per_volume)
        if len(train_examples) == 0:
            print(f'[{train_data_idx + 1} / {nb_train_data}] No positive samples annotation, skipped.')
            continue
        actual_sampled += len(train_examples)
        nb_files = np.math.ceil(len(train_examples) / args.samples_per_record)
        for i in range(nb_files):
            writer = tf.io.TFRecordWriter(os.path.join(train_root, f'data_{train_data_idx}_{i}.tfrecord'))
            j_begin = i * args.samples_per_record
            j_end = min((i + 1) * args.samples_per_record, len(train_examples))
            for j in range(j_begin, j_end):
                writer.write(train_examples[j].SerializeToString())
            writer.close()
        print(f'[{train_data_idx + 1} / {nb_train_data}] Already sampled:{actual_sampled}.')
    
    print('All examples generated.')

def make_train_examples(vol_name, seg_name, resample_size, patch_size, nb_samples_per_volume):

    bound = patch_size // 2
    actual_div_samples = nb_samples_per_volume // 2
    volume_nii = nibabel.load(vol_name)
    segmentation_nii = nibabel.load(seg_name)
    
    image = volume_nii.get_data()
    label = segmentation_nii.get_data()
    
    image = np.asarray(image).astype(np.float32)
    label = np.asarray(label).astype(np.uint8)

    image = resize(image, resample_size, order=1, preserve_range=True).astype(np.int16)
    label = resize(label, resample_size, order=0, preserve_range=True).astype(np.uint8)
    
    pos_core_index = np.argwhere(label[bound:-bound,bound:-bound,bound:-bound] == 1)
    neg_core_index = np.argwhere(label[bound:-bound,bound:-bound,bound:-bound] == 0)

    nb_pos_sample_cores = pos_core_index.shape[0]
    nb_neg_sample_cores = neg_core_index.shape[0]

    if nb_pos_sample_cores < actual_div_samples:
        print(f'Insufficient positive samples:{nb_pos_sample_cores}, consider a larger volume resample size.')
        nb_pos_samples = nb_pos_sample_cores
        nb_neg_samples = nb_samples_per_volume - nb_pos_samples
    else:
        nb_pos_samples = actual_div_samples
        nb_neg_samples = actual_div_samples

    np.random.shuffle(pos_core_index)
    np.random.shuffle(neg_core_index)

    sel_pos_index = pos_core_index[0:nb_pos_samples]
    sel_neg_index = neg_core_index[0:nb_neg_samples]
    examples = []
    for index in sel_pos_index:
        patch_image = image[index[0]:index[0]+patch_size,index[1]:index[1]+patch_size,index[2]:index[2]+patch_size]
        patch_label = label[index[0]:index[0]+patch_size,index[1]:index[1]+patch_size,index[2]:index[2]+patch_size]
        examples.append(tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(patch_image)])),
                    'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(patch_image.shape))),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(patch_label)])),
                    'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(patch_label.shape)))
                })))
    for index in sel_neg_index:
        patch_image = image[index[0]:index[0]+patch_size,index[1]:index[1]+patch_size,index[2]:index[2]+patch_size]
        patch_label = label[index[0]:index[0]+patch_size,index[1]:index[1]+patch_size,index[2]:index[2]+patch_size]
        examples.append(tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(patch_image)])),
                    'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(patch_image.shape))),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(patch_label)])),
                    'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(patch_label.shape)))
                })))
    
    return examples

def make_test_example(vol_name, seg_name, resample_size):

    volume_nii = nibabel.load(vol_name)
    segmentation_nii = nibabel.load(seg_name)
    aff = volume_nii.affine.astype(np.float32)
    
    image = volume_nii.get_data()
    label = segmentation_nii.get_data()
    
    image = np.asarray(image).astype(np.float32)
    label = np.asarray(label).astype(np.uint8)

    image = resize(image, resample_size, order=1, preserve_range=True).astype(np.int16)
    label = resize(label, resample_size, order=0, preserve_range=True).astype(np.uint8)
    
    return tf.train.Example(
    	features=tf.train.Features(
        	feature={
                'affine': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(aff)])),
                'affine_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(aff.shape))),
            	'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(image)])),
                'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(image.shape))),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(label)])),
                'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(label.shape)))
        	}
    	)
	)

if __name__ == "__main__":
    main()
