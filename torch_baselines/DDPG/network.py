import numpy as np
import jax.numpy as jnp
from torch.nn.modules.activation import Tanh
from torch_baselines.common.utils import get_flatten_size, visual_embedding
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_size,action_size,node=256,hidden_n=1,noisy=False,cnn_mode="normal"):
        super(Actor, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            visual_embedding(st,cnn_mode)
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
        
        self.linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, action_size[0]),
            nn.Tanh()
            ]
            )
        )

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        action = self.linear(cated)
        return action
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()
                
class Critic(nn.Module):
    def __init__(self,state_size,action_size,node=256,hidden_n=1,noisy=False,cnn_mode="normal"):
        super(Critic, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            visual_embedding(st,cnn_mode)
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        )) + action_size[0]
        
        self.linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node,1)
            ]
            )
        )

    def forward(self, xs,action):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)] + [action]
        cated = torch.cat(flats,dim=-1)
        q = self.q(cated)
        return q
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()