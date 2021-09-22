import torch_baselines.DQN.dqn
import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=256,Conv_option=False,Categorial_n=51,min=-200,max=200):
        super(Model, self).__init__()
        self.Categorial_n = Categorial_n
        self.action_size = action_size
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
            nn.Linear(node, action_size[0]*Categorial_n)
        )
        self.softmax = nn.Softmax(2)
        self.categorial_bar = torch.linspace(min,max,Categorial_n+1)
        self.mean_bar = ((self.categorial_bar[:-1] + self.categorial_bar[1:])/2.0).view(1,1,self.Categorial_n)
        print(self.categorial_bar)
        print(self.categorial_bar.shape)
        print(self.mean_bar)
        print(self.mean_bar.shape)

    def forward(self, xs):
        flat = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flat,dim=-1)
        x = self.linear(cated).view(-1,self.action_size,self.Categorial_n)
        x = self.softmax(x)
        return x
    
    def get_action(self,xs):
        with torch.no_grad():
            sm = self(xs)
            return (sm*self.mean_bar).mean(2).max(1)[1].view(-1,1).detach()