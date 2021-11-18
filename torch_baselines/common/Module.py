import numpy as np
from torch_baselines.common.utils import get_flatten_size, visual_embedding
import torch
import torch.nn as nn

class PreProcess(nn.Module):
    def __init__(self,state_size,cnn_mode="normal"):
        super(PreProcess, self).__init__()
        self.embedding = nn.ModuleList([
            visual_embedding(st,cnn_mode)
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        self.flatten_size = np.sum(
                            np.asarray(
                                [
                                get_flatten_size(pr,st)
                                for pr,st in zip(self.preembeddingprocess,state_size)
                                ]
                            ))
        
    def forward(self, xs):
        return torch.cat([pre(x) for pre,x in zip(self.embedding,xs)],dim=-1)