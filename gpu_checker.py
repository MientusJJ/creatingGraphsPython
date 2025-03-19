import tensorflow as tf
import functools

def requires_gpu(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError(" No GPU found! This function requires a GPU to run.")
        return func(*args, **kwargs)
    return wrapper
