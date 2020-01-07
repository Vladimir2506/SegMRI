from .model import U_Net, SimpleUNet, U_Net_2d
from .hybrid import HybridUNetX
import pdb

def get_model(cfg_model):

    model_name = cfg_model['name']
    model_kwargs = cfg_model['kwargs']
    model_name_ = model_name.lower()

    if model_name_ == 'unet':
        model = U_Net(**model_kwargs)
    elif model_name_ == 'simpleunet':
        model = SimpleUNet(**model_kwargs)
    elif model_name_ == 'unet2d':
        model = U_Net_2d(**model_kwargs)
    elif model_name_ == 'hybridunetx':
        model = HybridUNetX(**model_kwargs)
    else:
        raise NotImplementedError(f'{model_name} is not supported.')

    return model