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
    obs = [torch.Tensor(o,dtype=torch.float,device=device).permute(0,3,1,2) if len(o.shape) == 4 else torch.Tensor(o,dtype=torch.float,device=device) for o in obs]