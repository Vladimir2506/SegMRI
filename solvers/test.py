import tensorflow as tf
import numpy as np
import datetime
import time
import os

import pdb

from data import get_dataset
from models import get_model
from .optimizer import get_optimizer
from utils.metrics import DiceMeter, BinaryPRMeter, precision_recall, dice_coef
from utils.log_helper import Logger
from utils.vol_helper import decompose_vol2cube, compose_prob_cube2vol
from utils.nii_helper import save_pred_nii, save_label_nii, save_image_nii

class SolverTest(object):

    def __init__(self, config):
        
        self.config_data = config['data']
        self.config_model = config['model']
        self.config_solver = config['solver']

        self.test_steps = self.config_solver['test_steps']
        self.export = self.config_solver['export_results']
        self.hybrid = self.config_solver['hybrid']

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
        self.loss_meter = tf.metrics.Mean()

        log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.logger = Logger(os.path.join(self.log_root, f'{log_time}.log'))  
        self.batch_size = self.config_data['kwargs']['batch_size']
        self.input_shape = (self.batch_size, *self.config_model['input_shape'])
        self.model.build(self.input_shape)
        self.model.summary(print_fn=self.logger.info)
    
    def chop_forward(self, input_tensor):
       
        N, H, W, D, C = self.input_shape
        # z_layers = input_tensor.shape[3]
        pd = D // 2
        padded_tensor = tf.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (pd, pd), (0, 0)), 'symmetric')
        pz_layers = padded_tensor.shape[3]
        num_logits = np.zeros([1, H, W, pz_layers, 2])
        den_logits = np.zeros([1, H, W, pz_layers, 2])
        for d_idx in range(pz_layers):
            num_logits[:,:,:,d_idx:d_idx + D,:] += self.model(padded_tensor[:,:,:,d_idx:d_idx + D,:], training=False)
            den_logits[:,:,:,d_idx:d_idx + D,:] += 1.0
        logit_tensor = num_logits[:,:,:,pd:-pd,:] / den_logits[:,:,:,pd:-pd,:]
        return logit_tensor

    def save_model(self, step):

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_path = os.path.join(self.ckpt_root, f'step_{step:d}')
        checkpoint.write(ckpt_path)

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
            
            input_tensor = (image_tensor - tf.reduce_mean(image_tensor)) / tf.math.reduce_std(image_tensor)
            if self.hybrid:
                logit_tensor = self.chop_forward(input_tensor)
            else:
                inputs_list = decompose_vol2cube(tf.squeeze(input_tensor), self.batch_size, 64, 1, 4)
                logits_list = [self.model(x, training=False) for x in inputs_list]
                logit_tensor = compose_prob_cube2vol(logits_list, image_tensor.shape[1:4], self.batch_size, 64, 4, 2)
          
            pred_tensor = tf.nn.softmax(logit_tensor)[...,0:1]
            self.pr_meter.update_state(label_tensor, pred_tensor)
            self.dice_meter.update_state(label_tensor, pred_tensor)
            test_p, test_r = precision_recall(label_tensor, pred_tensor)
            test_dice = dice_coef(label_tensor, pred_tensor)
            
            self.logger.info(f'[Test {test_idx + 1}] Precision = {test_p:.4f}, Recall = {test_r:.4f}, Dice = {test_dice:.4f}.')

            if self.export:
                save_image_nii(image_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx}_img.nii'))
                save_pred_nii(pred_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx}_pred.nii'))
                save_label_nii(label_tensor, affine_tensor, os.path.join(self.result_root, f'{step}_{test_idx}_gt.nii'))

        test_p, test_r = self.pr_meter.result()
        test_dice = self.dice_meter.result()
        self.logger.info(f'[Total Average] Precision = {test_p:.4f}, Recall = {test_r:.4f}, Dice = {test_dice:.4f}.')