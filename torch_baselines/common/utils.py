import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import jax.numpy as jnp
from typing import Optional, List, Union
from torch import Tensor

def get_flatten_size(function,size):
    return function(torch.rand(*([1]+size))).data.shape[-1]

def convert_states(obs : List, device : torch.device):
    return [torch.tensor(o,dtype=torch.float32,device=device).permute(0,3,1,2) 
                if len(o.shape) == 4 else torch.tensor(o,dtype=torch.float32,device=device) for o in obs]

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * tau  + param.data * (1.0 - tau))