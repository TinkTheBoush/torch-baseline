import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import jax.numpy as jnp
from typing import Optional, List
from torch import Tensor

def get_flatten_size(function,size):
    return function(torch.rand(*([1]+size))).data.shape[-1]

def visual_embedding(st,mode="normal"):
    if mode == "normal":
        return  nn.Sequential(
                nn.Conv2d(st[0],32,kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
    elif mode == "minimum":
        return  nn.Sequential(
                nn.Conv2d(st[0],16,kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
    elif mode == "none":
        return  nn.Flatten()

def set_random_seed(seed):
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def convert_states(obs : List, device : torch.device, dtype=torch.float):
    return [torch.as_tensor(o,dtype=dtype,device=device).permute(0,3,1,2) 
                if len(o.shape) == 4 else torch.as_tensor(o,dtype=dtype,device=device) for o in obs]
    #return [torch.tensor(o,dtype=torch.float32,device=device).permute(0,3,1,2) 
    #            if len(o.shape) == 4 else torch.tensor(o,dtype=torch.float32,device=device) for o in obs]

@torch.jit.script
def hard_update(target : List[Tensor], source : List[Tensor]):
    for target_param, param in zip(target, source):
        target_param.data.copy_(param.data)

@torch.jit.script
def soft_update(target : List[Tensor], source : List[Tensor], tau : float):
    for target_param, param in zip(target, source):
        target_param.data.copy_(target_param.data * tau  + param.data * (1.0 - tau))