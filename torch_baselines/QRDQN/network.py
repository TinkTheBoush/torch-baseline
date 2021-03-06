import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size, visual_embedding
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=256,hidden_n=2,noisy=False,dualing=False,cnn_mode="normal",n_support = 200):
        super(Model, self).__init__()
        self.dualing = dualing
        self.noisy = noisy
        self.n_support = n_support
        self.action_size = action_size
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
        
        if not self.dualing:
            self.q_linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, action_size[0]*self.n_support)
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
            lin(node, action_size[0]*self.n_support)
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
            lin(node, self.n_support)
            ]
            )
            )
            
        self.noisy_param = []
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                self.noisy_param.append(m)
        

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        if not self.dualing:
            q = self.q_linear(cated).view(-1,self.action_size[0],self.n_support)
        else:
            a = self.advatage_linear(cated).view(-1,self.action_size[0],self.n_support)
            v = self.value_linear(cated).view(-1,1,self.n_support)
            q = v + a - a.mean(1, keepdim=True)
        return q
    
    def get_action(self,xs):
        with torch.no_grad():
            return self(xs).mean(2).max(1)[1].view(-1,1).detach().cpu().clone()
        
    def sample_noise(self):
        if not self.noisy:
            return
        for n in self.noisy_param:
            n.sample_noise()