import tensorflow as tf
import nibabel
import numpy as np

import pdb

def save_pred_nii(pred_tensor, affine, path):
    
    pred = np.asarray(pred_tensor)
    affine = np.asarray(affine).squeeze()
    
    pred = pred.squeeze()
    pred = np.clip(pred, 0.0, 1.0) * 32767.0
    pred = pred.astype(np.int16)

    pred_nii = nibabel.Nifti1Image(pred, affine)
    pred_nii.set_data_dtype(np.int16)
    nibabel.save(pred_nii, path)

def save_image_nii(img_tensor, affine, path):
    
    img = np.asarray(img_tensor).squeeze().astype(np.int16)
    affine = np.asarray(affine).squeeze()
    
    img_nii = nibabel.Nifti1Image(img, affine)
    img_nii.set_data_dtype(np.int16)
    nibabel.save(img_nii, path)

def save_label_nii(lbl_tensor, affine, path):
    
    lbl = np.asarray(lbl_tensor).squeeze().astype(np.uint8)
    affine = np.asarray(affine).squeeze()
    
    lbl_nii = nibabel.Nifti1Image(lbl, affine)
    lbl_nii.set_data_dtype(np.uint8)
    nibabel.save(lbl_nii, path)