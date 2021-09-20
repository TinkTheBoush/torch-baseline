import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp

def get_flatten_size(function,size):
    return function(torch.rand(*([1]+size))).data.shape[-1]