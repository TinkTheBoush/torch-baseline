import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=64,noisy=False,dualing=False,ModelOptions=None):
        super(Model, self).__init__()
        self.dualing = dualing
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[1],32,kernel_size=7,stride=3,padding=3,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=5,stride=2,padding=2,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
                nn.Flatten()
            )
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        if not self.dualing:
            self.q_linear = nn.Sequential(
                lin(flatten_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
        else:
            self.advatage_linear = nn.Sequential(
                lin(flatten_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
            
            self.value_linear = nn.Sequential(
                lin(flatten_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, 1)
            )

    def forward(self, xs):
        
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        if not self.dualing:
            q = self.q_linear(cated)
        else:
            a = self.advatage_linear(cated)
            v = self.value_linear(cated)
            q = v.view(-1,1) + (a - a.mean(-1,True))
        return q
    
    def get_action(self,xs):
        with torch.no_grad():
            return self(xs).max(-1)[1].view(-1,1).detach().cpu().clone()
        
    def samepl_noise(self):
        if not self.noisy:
            return
        for idx,m in enumerate(self.modules()):
            if isinstance(m,NoisyLinear):
                m.sample_noise()