import tensorflow as tf
import numpy as np
import datetime
import time
import os

import nibabel as nib
import matplotlib.pyplot as plt

import pdb

from data import get_dataset
from models import get_model
from .optimizer import get_optimizer
from utils.metrics import DiceMeter, BinaryPRMeter, precision_recall, dice_coef
from utils.log_helper import Logger
from utils.nii_helper import save_label_nii, save_image_nii, save_pred_nii

class SolverTest_2d(object):

    def __init__(self, config):
        
        self.config_data = config['data']
        self.config_model = config['model']
        self.config_solver = config['solver']

        self.test_steps = self.config_solver['test_steps']
        self.export = self.config_solver['export_results']
        self.dataset = get_dataset(self.config_data)
        
        self.exp_root = self.config_solver['exp_root']
        if not os.path.exists(self.exp_root):
            os.makedirs(self.exp_root)
        self.ckpt_root = os.path.join(self.exp_root, 'models')
        if not os.path.exists(self.ckpt_root):
            os.makedirs(self.ckpt_root)
        self.log_root = os.path.join(self.exp_root, 'logs')
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.result_root = os.path.join(self.exp_root, 'results')
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)

        self.model = get_model(self.config_model)
        self.optimizer = get_optimizer(self.config_solver)
        
        self.pr_meter = BinaryPRMeter()
        self.dice_meter = DiceMeter()

        log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.logger = Logger(os.path.join(self.log_root, f'{log_time}.log'))  

        self.input_shape = (self.config_data['kwargs']['batch_size'], *self.config_model['input_shape'])
        self.model.build(self.input_shape)
        self.model.summary(print_fn=self.logger.info)  

    # @tf.function
    def chop_forward(self, input_tensor, gt):

        # pdb.set_trace()
        converted_tensor = input_tensor[0,...]
        converted_tensor = tf.transpose(converted_tensor, (2, 0, 1, 3))
        batch_size = self.input_shape[0]
        z_layers = input_tensor.shape[3]
        nb_batches = tf.math.ceil(z_layers / batch_size)
        logits = []
        for z in range(nb_batches):
            begin = tf.cast(z * batch_size, tf.int32)
            end = tf.cast(tf.minimum((z + 1) * batch_size, z_layers), tf.int32)
            sliced_tensor = converted_tensor[begin:end,...]
            sliced_tensor = (sliced_tensor - tf.reduce_mean(sliced_tensor)) / tf.math.reduce_std(sliced_tensor)
            logit_sliced_tensor = self.model(sliced_tensor, training=False)
            logits.append(logit_sliced_tensor)
        logit_tensor = tf.concat(logits, axis=0)
        logit_tensor = tf.transpose(logit_tensor, (1, 2, 0, 3))
        logit_tensor = tf.expand_dims(logit_tensor, 0)
        
        return logit_tensor

    def load_model(self, step):

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_path = os.path.join(self.ckpt_root, f'step_{step:d}')
        checkpoint.restore(ckpt_path).expect_partial()

    def run(self):

        self.logger.info('Testing started.')
        # start_tick = time.time()
        test_loader = self.dataset.get_loader(training=False)

        step = self.test_steps
        self.load_model(step)
        self.logger.info(f'[Step {step:d}] Model Loaded.')

        for test_idx, (image_tensor, label_tensor, affine_tensor) in enumerate(test_loader):

            # pdb.set_trace()
            logit_tensor = self.chop_forward(image_tensor, label_tensor)
            pred_tensor = tf.nn.softmax(logit_tensor)[...,0:1]

            print(f'positive predictions:{tf.math.count_nonzero(pred_tensor > 0.5).numpy()}.')
            
            self.pr_meter.update_state(label_tensor, pred_tensor)
            self.dice_meter.update_state(label_tensor, pred_tensor)
            test_p, test_r = precision_recall(label_tensor, pred_tensor)
            test_dice = dice_coef(label_tensor, pred_tensor)

            if self.export:
                
                save_image_nii(image_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx + 1}_img.nii'))
                save_pred_nii(pred_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx + 1}_pred.nii'))
                save_label_nii(label_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx + 1}_gt.nii'))

            self.logger.info(f'[Test {test_idx + 1}] Precision = {test_p:.4f}, Recall = {test_r:.4f}, Dice = {test_dice:.4f}.')
            

        test_p, test_r = self.pr_meter.result()
        test_dice = self.dice_meter.result()
        self.logger.info(f'[Total Average] Precision = {test_p:.4f}, Recall = {test_r:.4f}, Dice = {test_dice:.4f}.')