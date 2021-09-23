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
        return (weight * F.mse_loss(input, target, reduction='none').squeeze()).mean(-1)

class WeightedHuber(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(WeightedHuber, self).__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor, weight : Tensor) -> Tensor:
        return (weight * F.smooth_l1_loss(input, target, reduction='none', beta=self.beta).squeeze()).mean(-1)
    
class Categorial51Loss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', categorial_bar = None) -> None:
        super(Categorial51Loss, self).__init__(size_average, reduce, reduction)
        self.categorial_bar = categorial_bar
        self.bar_start = categorial_bar[:-1]
        self.bar_end = categorial_bar[1:]

    def forward(self, input_distribution: Tensor, next_distribution: Tensor, next_categorial_bar: Tensor) -> Tensor:
        with torch.no_grad():
            Tz = next_categorial_bar.clamp(self.categorial_bar[0], self.categorial_bar[-1])
            C51_b = (Tz - C51_vmin) / C51_delta
            C51_L = C51_b.floor().to(torch.int64)
            C51_U = C51_b.ceil().to(torch.int64)
            C51_L[ (C51_U > 0)               * (C51_L == C51_U)] -= 1
            C51_U[ (C51_L < (C51_atoms - 1)) * (C51_L == C51_U)] += 1
            target_distribution = next_distribution
        return F.binary_cross_entropy_with_logits(input_distribution,target_distribution)

def categorial_loss(input_distribution,next_distribution,categorial_bar,next_categorial_bar):
    bar_start = categorial_bar[:-1]
    bar_end = categorial_bar[1:]
    target_distribution = next_distribution
    x = target_distribution - input_distribution
    loss = -(x*torch.log(x)).mean(1).mean(0)
    return loss