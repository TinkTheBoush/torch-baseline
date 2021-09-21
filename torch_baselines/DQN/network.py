import torch_baselines.DQN.dqn
import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=512,Conv_option=False):
        super(Model, self).__init__()
        self.preprocess = [
            nn.Sequential(
                nn.Conv2d(3,32,kernel_size=7,stride=3,padding=3,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=5,stride=2,padding=2,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
                nn.Flatten()
            )
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ]
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        self.linear = nn.Sequential(
            nn.Linear(flatten_size,node),
            nn.ReLU(),
            nn.Linear(node,node),
            nn.ReLU(),
            nn.Linear(node, action_size[0])
        )

    def forward(self, xs):
        flat = [pre(x) for pre,x in zip(self.preprocess,xs)]
        #print(flat)
        cated = torch.cat(flat,dim=-1)
        x = self.linear(cated)
        return x
    
    def get_action(self,xs):
        with torch.no_grad():
            return self(xs).max(-1)[1]