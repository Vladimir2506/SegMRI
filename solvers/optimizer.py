import tensorflow as tf

def get_optimizer(cfg_solver):

    optimizer_name = cfg_solver['optimizer']['name']
    optimizer_kwargs = cfg_solver['optimizer']['kwargs']
    optimizer_name_ = optimizer_name.lower()
    lr_init = cfg_solver['lr']
    sch_ = cfg_solver.get('lr_scheduler', None)
    if sch_ is not None:
        lr_scheduler_name = cfg_solver['lr_scheduler']['name']
        lr_scheduler_name_ = lr_scheduler_name.lower()
        lr_scheduler_kwargs = cfg_solver['lr_scheduler']['kwargs']
        if lr_scheduler_name_ == 'exponential':
            lr_scheduled = tf.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr_init,
                **lr_scheduler_kwargs
            )
        else:
            raise NotImplementedError(f'{lr_scheduler_name} is not supported.')
    else:
        lr_scheduled = lr_init
    
    if optimizer_name_ == 'adam':
        optimizer = tf.optimizers.Adam(
            learning_rate=lr_scheduled,
            **optimizer_kwargs
        )
    elif optimizer_name_ == 'sgd':
        optimizer = tf.optimizers.SGD(
            learning_rate=lr_scheduled, 
            **optimizer_kwargs
        )
    elif optimizer_name_ == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(
            learning_rate=lr_scheduled,
            **optimizer_kwargs
        )
    else:
        raise NotImplementedError(f'{optimizer_name} is not supported.')

    return optimizer