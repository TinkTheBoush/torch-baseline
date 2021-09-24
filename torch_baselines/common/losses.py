import torch
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss


from torch import Tensor
from typing import Callable, Optional

class MSELosses(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELosses, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction='none').squeeze()

class HuberLosses(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(HuberLosses, self).__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction='none', beta=self.beta).squeeze() #.mean(-1)
    
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

class QRHuberLosses(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0, support_size = 64) -> None:
        super(QRHuberLosses, self).__init__(size_average, reduce, reduction)
        self.beta = beta
        self.support_size = support_size

    def forward(self, theta_loss_tile: Tensor, logit_valid_tile: Tensor, quantile: Tensor) -> Tensor:
        quantile = quantile.view(1,1,self.support_size)
        huber = F.smooth_l1_loss(theta_loss_tile, logit_valid_tile, reduction='none', beta=self.beta)
        with torch.no_grad():
            bellman_errors = logit_valid_tile - theta_loss_tile
            mul = torch.abs(quantile - (bellman_errors < 0).float())
        return (mul*huber).sum(1).mean(1) #.mean(-1)