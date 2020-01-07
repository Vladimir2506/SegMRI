# MR Image Segmentation

> 2019 ML Course Project

## Confidential Notification

- **NOTE**: This repository contains **confidential data**, which are **not permitted** to release to public.

- This repository should only contain codes and models, please check your contents whenever committing.

---

## Requirements & Environment Settings

- This project require these packages.

    - `python=3.7`
    - `tensorflow=2.0.0`
    - `pyyaml`
    - `numpy`
    - `nibabel`
    - `scikit-image`
    - `scipy`

- To setup the environment, please follow these steps.

    0. If you are willing to use GPU, please follow these instructions to setup CUDA environments https://www.tensorflow.org/install/gpu?hl=zh_cn.

    1. Set channels to Tsinghua Mirrors, please follow these instructions https://tuna.moe/oh-my-tuna/

    2. Use Anaconda to create a new environment

    ```
    > conda create -n tf python=3.7
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
    
    2. Run `gen_mri_**.sh` as follows, to generate datasets in three settings.

    ```
    > sh -x gen_mri_data.sh
    > sh -x gen_mri_2d_data.sh
    > hs -x gen_mri_hybrid_data.sh
    ```

- Please **DO NOT COMMIT** any content of this dataset for its confidentiality. 

### LiTS dataset (public)

- We gave up using this dataset as a pretraining option.

## Directory Structure

```
.
├── configs
│   ├── HybridUnetX_MRI_artery.yaml
│   ├── HybridUnetX_MRI_portal.yaml
│   ├── SimpleUNet_MRI_artery.yaml
│   ├── SimpleUNet_MRI_portal.yaml
│   ├── UNet2d_MRI_artery.yaml
│   └── UNet2d_MRI_portal.yaml
├── data
│   ├── augmentation.py
│   ├── dataset_2d.py
│   ├── dataset.py
│   ├── __init__.py
│   └── __pycache__
├── experiments
│   ├── hybrid_mri_artery
│   ├── hybrid_mri_portal
│   ├── simpleunet_mri_artery
│   ├── simpleunet_mri_portal
│   ├── unet2d_mri_artery
│   └── unet2d_mri_portal
├── gen_mri_2d_data.sh
├── gen_mri_data.sh
├── gen_mri_hybrid_data.sh
├── main.py
├── models
│   ├── attention_blocks.py
│   ├── hybrid.py
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   └── residual_blocks.py
├── readme.md
├── report.pdf
├── solvers
│   ├── __init__.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── __pycache__
│   ├── test_2d.py
│   ├── test.py
│   ├── train_2d.py
│   └── train.py
├── test_mri_artery_2d.sh
├── test_mri_artery_hybrid.sh
├── test_mri_artery.sh
├── test_mri_portal_2d.sh
├── test_mri_portal_hybrid.sh
├── test_mri_portal.sh
├── tools
│   ├── make_mri_2d_dataset.py
│   ├── make_mri_dataset.py
│   └── make_mri_hybrid_dataset.py
├── train_mri_artery_2d.sh
├── train_mri_artery_hybrid.sh
├── train_mri_artery.sh
├── train_mri_portal_2d.sh
├── train_mri_portal_hybrid.sh
├── train_mri_portal.sh
└── utils
    ├── __init__.py
    ├── log_helper.py
    ├── metrics.py
    ├── nii_helper.py
    ├── __pycache__
    └── vol_helper.py
```

- `configs` contains all configuration of models stated in the report, including all hyperparameters, you can adjust some value to define your own model.

- `data` contains datasets and dataloaders implementation, and there are also some data augmentation operations. 

- `models` contains codes of architectures of all models.

- `solvers` contains codes for training / testing process in 2D and 3D settings.

- `tools` contains the `.tfrecord` dataset generator.

- `utils` contains some tiny utility codes in use.

- `main.py` is the **main entrance** of this program, follow the `train_*.sh/test_*.sh` to start.

## Usage

### Reproduce all results

- Run `test_*.sh`, then all quantitative results would be shown in shell, and the qualitative results would be stored in the corresponding experiment subdir, stored in `results` folder, with `*.nii` format.

### Train your own model

- Configure the parameters in `configs`, and then run `train_*.sh`.

