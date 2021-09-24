import torch_baselines.DQN.dqn
import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=64,Conv_option=False,quantile_n = 32):
        super(Model, self).__init__()
        self.quantile_n = quantile_n
        self.action_size = action_size
        self.preprocess = nn.ModuleList([
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
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        self.q_linear = nn.Sequential(
            nn.Linear(flatten_size,node),
            nn.ReLU(),
            nn.Linear(node,node),
            nn.ReLU(),
            nn.Linear(node, action_size[0]*quantile_n)
        )
        

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        x = self.q_linear(cated).view(-1,self.action_size,self.quantile_n)
        return x
    
    def get_action(self,xs):
        with torch.no_grad():
            return self(xs).mean(2).max(1)[1].view(-1,1).detach().cpu().clone()
        
class Dualing_Model(nn.Module):
    def __init__(self,state_size,action_size,node=64,Conv_option=False,quantile_n = 32):
        super(Dualing_Model, self).__init__()
        self.quantile_n = quantile_n
        self.action_size = action_size
        self.preprocess = nn.ModuleList([
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
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        self.advatage_linear = nn.Sequential(
            nn.Linear(flatten_size,node),
            nn.ReLU(),
            nn.Linear(node,node),
            nn.ReLU(),
            nn.Linear(node, action_size[0]*quantile_n)
        )
        
        self.value_linear = nn.Sequential(
            nn.Linear(flatten_size,node),
            nn.ReLU(),
            nn.Linear(node,node),
            nn.ReLU(),
            nn.Linear(node, quantile_n)
        )

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        a = self.advatage_linear(cated).view(-1,self.action_size,self.quantile_n)
        v = self.value_linear(cated).view(-1,self.quantile_n)
        q = v + (a - a.mean(1,True))
        return q
    
    def get_action(self,xs):
        with torch.no_grad():
            return self(xs).mean(2).max(1)[1].view(-1,1).detach().cpu().clone()