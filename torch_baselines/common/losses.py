import torch
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss


from torch import Tensor
from typing import Callable, Optional

class WeightedMSELoss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, weight : Tensor) -> Tensor:
        
        return (F.mse_loss(input, target, reduction=False) * weight).mean(-1)

class WeightedHuber(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(WeightedHuber, self).__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor, weight : Tensor) -> Tensor:
        return (F.smooth_l1_loss(input, target, reduction=False, beta=self.beta)* weight).mean(-1) 

def categorial_loss(input_distribution,next_distribution,categorial_bar,next_categorial_bar):
    bar_start = categorial_bar[:-1]
    bar_end = categorial_bar[1:]
    target_distribution = next_distribution
    x = target_distribution - input_distribution
    loss = -(x*torch.log(x)).mean(1).mean(0)
    return loss