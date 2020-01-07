import tensorflow as tf
import numpy as np
import argparse
import yaml
import os

from solvers import get_solver

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to configuration file to run.')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train_2d', 'test_2d'], help='Mode of solver to run.')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Whether make reproducible results.')
    parser.add_argument('--gpu_id', type=str, help='Specify devices numbers to run.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id # Set GPU visible to this program
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Block a lot of information output by TensorFlow
    
    # Fixed random seed and set deterministic mode in cuDNN backend
    if args.deterministic:
        print('Deterministic mode.')
        os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
        tf.random.set_seed(42)
        np.random.seed(42)

    # Load configuration from YAML
    with open(args.config, 'r') as yaml_f:
        configs = yaml.safe_load(yaml_f)
    
    # Set tensorflow GPU growth
    devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in devices:
        tf.config.experimental.set_memory_growth(dev, True)

    # Create a solver and run
    solver = get_solver(args.mode, configs)
    solver.run()

if __name__ == "__main__":
    main()
    