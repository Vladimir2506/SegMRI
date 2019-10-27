# MR Image Segmentation

> 2019 ML Course Project


## Confidential Notification

- **NOTE**: This repository contains **confidential data**, which are **not permitted** to release to public.

- This repository should only contain codes and models, please check your contents whenever committing.

---

## Requirements & Environment Settings

- This project require these packages.

    - `Python=3.6`
    - `Tensorflow=2.0.0`
    - `PyYAML`
    - `Numpy`
    - `matplotlib`
    - `nibabel`
    - `scikit-image`

- To setup the environment, please follow these steps.

    0. If you are willing to use GPU, please follow these instructions to setup CUDA environments https://www.tensorflow.org/install/gpu?hl=zh_cn.

    1. Set channels to Tsinghua Mirrors, please follow these instructions https://tuna.moe/oh-my-tuna/

    2. Use Anaconda to create a new environment

    ```
    > conda create -n tf python=3.6
    > conda activate tf
    # if use gpu
    > conda install tensorflow-gpu=2.0
    # else 
    > conda install tensorflow=2.0
    > conda install matplotlib pyyaml scikit-image
    > pip install nibabel
    ```

    3. Environment setup is done.

## Datasets Preparation

### In-house MR image dataset (confidential)

- To prepare tfrecord dataset, please follow this steps.

    1. Extract raw data in *.nii* format and store them like this tree structure.

    ```
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
    ```
    
    2. Run `tools/make_dataset.py` as follows.

    ```
    > python tools/make_dataset.py \
    >   --raw_data_root path/to/raw_root \
    >   --record_root path/to/tfrecords \
    >   --max_samples 1
    ```

- Please **DO NOT COMMIT** any content of this dataset for its confidentiality. 