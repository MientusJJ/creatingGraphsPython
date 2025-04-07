import tensorflow as tf
import functools

import torch


def requires_gpu(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found! This function requires a GPU to run.")
        return func(*args, **kwargs)

    return wrapper
