from .train import SolverTrain
from .test import SolverTest
from .train_2d import SolverTrain_2d
from .test_2d import SolverTest_2d

def get_solver(mode, config):

    mode_ = mode.lower()
    
    if mode_ == 'train':
        return SolverTrain(config)
    elif mode_ == 'test':
        return SolverTest(config)
    elif mode_ == 'train_2d':
        return SolverTrain_2d(config)
    elif mode_ == 'test_2d':
        return SolverTest_2d(config)
    else:
        raise NotImplementedError(f'Solver [{mode}] is not supported.')