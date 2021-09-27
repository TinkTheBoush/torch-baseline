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
    

class CategorialDistributionLoss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', batch_size = None, categorial_bar = None, categorial_bar_n = 51, delta = None) -> None:
        super(CategorialDistributionLoss, self).__init__(size_average, reduce, reduction)
        self.categorial_bar = categorial_bar
        self.min = categorial_bar[0][0]
        self.max = categorial_bar[0][-1]
        self.delta = delta
        self.categorial_bar_n = categorial_bar_n
        self.batch_size = batch_size
        offset = torch.linspace(0, (self.batch_size - 1) * categorial_bar_n, self.batch_size)
        offset = offset.unsqueeze(dim=1) 
        self.offset = offset.expand(self.batch_size, categorial_bar_n) # I believe this is to(device)


    def forward(self, input_distribution: Tensor, next_distribution: Tensor, next_categorial_bar: Tensor) -> Tensor:
        with torch.no_grad():
            Tz = next_categorial_bar.clamp(self.min, self.max)
            C51_b = (Tz - self.min) / self.delta
            C51_L = C51_b.floor().int()
            C51_U = C51_b.ceil().int()
            C51_L[ (C51_U > 0)               * (C51_L == C51_U)] -= 1
            C51_U[ (C51_L < (self.categorial_bar_n - 1)) * (C51_L == C51_U)] += 1
            self.offset = self.offset.to(next_distribution).int()
            target_distribution = input_distribution.new_zeros(self.batch_size, self.categorial_bar_n) # Returns a Tensor of size size filled with 0. same dtype
            target_distribution.view(-1).index_add_(0, (C51_L + self.offset).view(-1), (next_distribution * (C51_U.float() - C51_b)).view(-1))
            target_distribution.view(-1).index_add_(0, (C51_U + self.offset).view(-1), (next_distribution * (C51_b - C51_L.float())).view(-1))
        return -(target_distribution * input_distribution.log()).sum(-1)
        #F.binary_cross_entropy_with_logits(input_distribution,target_distribution, reduction='none').sum(-1)
        #-(target_distribution * input_distribution.log()).sum(-1)

class QRHuberLosses(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0, support_size = 64) -> None:
        super(QRHuberLosses, self).__init__(size_average, reduce, reduction)
        self.beta = beta
        self.support_size = support_size

    def forward(self, theta_loss_tile: Tensor, logit_valid_tile: Tensor, quantile: Tensor) -> Tensor:
        huber = F.smooth_l1_loss(theta_loss_tile, logit_valid_tile, reduction='none', beta=self.beta)
        with torch.no_grad():
            bellman_errors = logit_valid_tile - theta_loss_tile
            mul = torch.abs(quantile - (bellman_errors < 0).float())
        return (mul*huber).sum(1).mean(1) #.mean(-1)