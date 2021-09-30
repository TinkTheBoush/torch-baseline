import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=256,noisy=False,dualing=False,ModelOptions=None):
        super(Model, self).__init__()
        self.dualing = dualing
        self.noisy = noisy
        self.action_size = action_size
        self.node = node
        
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],32,kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=3,stride=1),
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
        
        self.preprocess.flatten_size = flatten_size
        
        self.embedding_size = np.maximum(flatten_size, 256)
        
        self.state_embedding = nn.Sequential(
            lin(flatten_size,self.embedding_size),
            nn.ReLU()
        )
        
        self.register_buffer('pi_mtx', torch.from_numpy(np.expand_dims(np.pi * np.arange(0, 128,dtype=np.float32), axis=0))) # for non updating constant values
        self.quantile_embedding = nn.Sequential(
            lin(128,self.embedding_size),
            nn.ReLU()
        )
        
        if not self.dualing:
            self.q_linear = nn.Sequential(
                lin(self.embedding_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
        else:
            self.advatage_linear = nn.Sequential(
                lin(self.embedding_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
            
            self.value_linear = nn.Sequential(
                lin(self.embedding_size,node),
                nn.ReLU(),
                lin(node,node),
                nn.ReLU(),
                lin(node, 1)
            )
        

    def forward(self, xs, quantile):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        n_support = quantile.shape[1]
        
        state_embed = self.state_embedding(cated).unsqueeze(2).repeat_interleave(n_support, dim=2).view(-1,self.embedding_size)
        costau = quantile.view(-1,1)*self.pi_mtx
        quantile_embed = self.quantile_embedding(costau)
        print(state_embed.shape)
        print(quantile_embed.shape)
        
        mul_embed = torch.multiply(state_embed,quantile_embed)
        
        if not self.dualing:
            q = self.q_linear(mul_embed).view(-1,n_support,self.action_size[0])
        else:
            a = self.advatage_linear(mul_embed).view(-1,n_support,self.action_size[0])
            v = self.value_linear(mul_embed).view(-1,n_support,1)
            q = v + a - a.mean(1, keepdim=True)
        return q
                
class Qunatile_Maker(nn.Module):
    def __init__(self,n_support = 64):
        super(Qunatile_Maker, self).__init__()
        self.n_support = n_support
        self.dummy_param = nn.Parameter(torch.empty(0)) #for auto device sinc!
    
    def forward(self, buffer_size):
        return torch.rand([buffer_size,self.n_support],dtype=torch.float32,device=self.dummy_param.device)