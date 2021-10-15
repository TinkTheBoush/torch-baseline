import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=128,hidden_n=2,noisy=False,dualing=False,ModelOptions=None,categorial_bar=None,categorial_bar_n=51):
        super(Model, self).__init__()
        self.dualing = dualing
        self.noisy = noisy
        self.categorial_bar = categorial_bar
        self.categorial_bar_n = categorial_bar_n
        self.action_size = action_size
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],32,kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),
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
        
        if not self.dualing:
            self.q_linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, action_size[0]*self.categorial_bar_n)
            ]
            )
            )
        else:
            self.advatage_linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, action_size[0]*self.categorial_bar_n)
            ]
            )
            )
            
            self.value_linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, action_size[0]*self.categorial_bar_n)
            ]
            )
            )
            
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xs):
        
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        if not self.dualing:
            q = self.q_linear(cated).view(-1,self.action_size[0],self.categorial_bar_n)
        else:
            a = self.advatage_linear(cated).view(-1,self.action_size[0],self.categorial_bar_n)
            v = self.value_linear(cated).view(-1,1,self.categorial_bar_n)
            q = v + (a - a.mean(1,True))
        return self.softmax(q)
    
    def get_action(self,xs):
        with torch.no_grad():
            return (self(xs)*self.categorial_bar).sum(2).max(1)[1].view(-1,1).detach().cpu().clone()
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()