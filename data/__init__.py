from .dataset import Dataset
from .dataset_2d import Dataset_2d

def get_dataset(cfg_data):

    name, kwargs = cfg_data['name'], cfg_data['kwargs']
    name_ = name.lower()
    if name_ in ['lits', 'mri']:
        return Dataset(**kwargs)
    elif name_ in ['mri2d']:
        return Dataset_2d(**kwargs)
    else:
        raise NotImplementedError(f'Dataset [{name}] is not supported.')
