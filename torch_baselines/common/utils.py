import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import jax.numpy as jnp
from typing import Optional, List
import numpy.typing as npt
from torch import Tensor

def get_flatten_size(function,size):
    return function(torch.rand(*([1]+size))).data.shape[-1]

def set_random_seed(seed):
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

#def convert_states(obs : List[npt.NDArray[np.float32]], device : torch.device):
@torch.jit.script
def convert_states(obs, device : torch.device):
    return [torch.tensor(o,dtype=torch.float32,device=device).permute(0,3,1,2) 
                if len(o.shape) == 4 else torch.tensor(o,dtype=torch.float32,device=device) for o in obs]

@torch.jit.script
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

@torch.jit.script
def soft_update(target, source, tau : float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * tau  + param.data * (1.0 - tau))