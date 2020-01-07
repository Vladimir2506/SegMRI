import tensorflow as tf
import time
import datetime
import os

import pdb

import matplotlib.pyplot as plt

from data import get_dataset
from models import get_model
from .losses import get_loss
from .optimizer import get_optimizer
from utils.metrics import DiceMeter, BinaryPRMeter
from utils.log_helper import Logger

class SolverTrain_2d(object):

    def __init__(self, config):
        
        self.config_data = config['data']
        self.config_model = config['model']
        self.config_solver = config['solver']

        self.max_steps = self.config_solver['max_steps']
        self.save_steps = self.config_solver['save_steps']
        self.log_steps = self.config_solver['log_steps']
        self.resume_steps = self.config_solver['resume_steps']

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

        self.model = get_model(self.config_model)
        # self.loss = BalancedLoss(self.config_solver)
        self.loss = get_loss(self.config_solver['loss'])
        self.optimizer = get_optimizer(self.config_solver)
        
        self.pr_meter = BinaryPRMeter()
        self.dice_meter = DiceMeter()
        self.loss_meter = tf.metrics.Mean()

        log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.logger = Logger(os.path.join(self.log_root, f'{log_time}.log'))  

        self.input_shape = (self.config_data['kwargs']['batch_size'], *self.config_model['input_shape'])
        # pdb.set_trace()
        self.model.build(self.input_shape)
        self.model.summary(print_fn=self.logger.info)      
        
    @tf.function
    def train_step(self, image_tensor, label_tensor):
        
        with tf.GradientTape() as tape:
            image_tensor = (image_tensor - tf.reduce_mean(image_tensor)) / tf.math.reduce_std(image_tensor)
            logit_tensor = self.model(image_tensor, training=True)
            train_loss_value = self.loss(label_tensor, logit_tensor)
            
        grads = tape.gradient(train_loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        pred_tensor = tf.nn.softmax(logit_tensor)[...,0:1]
        return train_loss_value, pred_tensor

    def save_model(self, step):

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_path = os.path.join(self.ckpt_root, f'step_{step:d}')
        checkpoint.write(ckpt_path)

    def load_model(self, step):

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_path = os.path.join(self.ckpt_root, f'step_{step:d}')
        checkpoint.restore(ckpt_path).expect_partial()

    def run(self):
        
        self.logger.info('Training started.')
        start_tick = time.time()
        train_loader = self.dataset.get_loader(training=True)

        if self.resume_steps > 0:
            step = self.resume_steps
            self.load_model(step)
            self.logger.info(f'[Step {step}] Checkpoint resumed.')
        else:
            step = 0
            self.logger.info('Train from scratch.')

        step += 1

        for image_tensor, label_tensor in train_loader:

            train_loss_value, pred_tensor = self.train_step(image_tensor, label_tensor)
            self.dice_meter.update_state(label_tensor, pred_tensor)
            self.pr_meter.update_state(label_tensor, pred_tensor)
            self.loss_meter.update_state(train_loss_value)

            if step % self.log_steps == 0:
                train_p, train_r = self.pr_meter.result()
                train_dice = self.dice_meter.result()
                train_loss_value = self.loss_meter.result()
                self.pr_meter.reset_states()
                self.dice_meter.reset_states()
                self.loss_meter.reset_states()
                elapsed_time = time.time() - start_tick
                et = str(datetime.timedelta(seconds=elapsed_time))[:-7]
                cur_lr = self.optimizer._decayed_lr(tf.float32)

                self.logger.info(f'[{et} Step {step:d} / {self.max_steps}] Training loss = {train_loss_value:.6f}, lr = {cur_lr:.6f}, Precision = {train_p:.4f}, Recall = {train_r:.4f}, Dice = {train_dice:.4f}.')
            
            if self.save_steps > 0 and step % self.save_steps == 0:
                self.save_model(step)
                self.logger.info(f'[Step {step}] checkpoint saved.')
            
            if step >= self.max_steps:
                break

            step += 1
            
        
        self.logger.info('Training finished.')
