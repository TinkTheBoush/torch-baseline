import numpy as np
import jax.numpy as jnp
from torch.nn.modules.activation import Tanh
from torch_baselines.common.utils import get_flatten_size
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_size,action_size,node=256,noisy=False,ModelOptions=None):
        super(Actor, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],16,kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=3,stride=1),
                nn.ReLU(),
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
        
        self.linear = nn.Sequential(
            lin(flatten_size,node),
            nn.ReLU(),
            lin(node,node),
            nn.ReLU(),
            lin(node,node),
            nn.ReLU(),
            lin(node, action_size[0]),
            nn.Tanh()
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
    def __init__(self,state_size,action_size,node=256,noisy=False,ModelOptions=None):
        super(Critic, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],16,kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=3,stride=1),
                nn.ReLU(),
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
                        )) + action_size[0]
        self.preprocess.flatten_size = flatten_size
        
        self.embedding_size = np.maximum(flatten_size, node)
        
        if not (flatten_size == self.embedding_size):
            self.state_embedding = nn.Sequential(
                lin(flatten_size,self.embedding_size),
                nn.ReLU()
            )
        
        self.register_buffer('pi_mtx', torch.from_numpy(np.expand_dims(np.pi * np.arange(0, 128,dtype=np.float32), axis=0))) # for non updating constant values
        self.quantile_embedding = nn.Sequential(
            lin(128,self.embedding_size),
            nn.ReLU()
        )
        
        self.q1 = nn.Sequential(
            lin(self.embedding_size,node),
            nn.ReLU(),
            lin(node,node),
            nn.ReLU(),
            lin(node, 1)
        )
        
        self.q2 = nn.Sequential(
            lin(self.embedding_size,node),
            nn.ReLU(),
            lin(node,node),
            nn.ReLU(),
            lin(node, 1)
        )

    def forward(self, xs,action,quantile):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)] + [action]
        cated = torch.cat(flats,dim=-1)
        n_support = quantile.shape[1]
        if self.preprocess.flatten_size == self.embedding_size:
            state_embed = cated
        else:
            state_embed = self.state_embedding(cated)
        state_embed = state_embed.unsqueeze(1).repeat_interleave(n_support, dim=1).view(-1,self.embedding_size)
        costau = quantile.view(-1,1)*self.pi_mtx
        quantile_embed = self.quantile_embedding(costau)
        mul_embed = torch.multiply(state_embed,quantile_embed)
        q1 = self.q1(mul_embed).view(-1,n_support)
        q2 = self.q2(mul_embed).view(-1,n_support)
        return q1, q2
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()
                
class Qunatile_Maker(nn.Module):
    def __init__(self,n_support = 64):
        super(Qunatile_Maker, self).__init__()
        self.n_support = n_support
        self.dummy_param = nn.Parameter(torch.empty(0)) #for auto device sinc!
    
    def forward(self, buffer_size):
        return torch.rand([buffer_size,self.n_support],dtype=torch.float32,device=self.dummy_param.device)