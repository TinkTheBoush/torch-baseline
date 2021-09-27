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
        offset = torch.linspace(-1, (self.batch_size - 1) * categorial_bar_n - 1, self.batch_size)
        offset = offset.unsqueeze(dim=1) 
        self.offset = offset.expand(self.batch_size, categorial_bar_n) # I believe this is to(device)


    def forward(self, input_distribution: Tensor, next_distribution: Tensor, next_categorial_bar: Tensor) -> Tensor:
        with torch.no_grad():
            Tz = next_categorial_bar.clamp(self.categorial_bar[0], self.categorial_bar[-1])
            Tz = Tz.clamp(self.min, self.max)
            C51_b = (Tz - self.min) / self.delta
            C51_L = C51_b.floor().int()
            C51_U = C51_b.ceil().int()
            C51_L[ (C51_U > 0) * (C51_L == C51_U)] -= 1
            C51_U[ (C51_L < (self.categorial_bar_n - 1)) * (C51_L == C51_U)] += 1
            self.offset = self.offset.to(next_distribution).int()
            target_distribution = input_distribution.new_zeros(self.batch_size, self.categorial_bar_n) # Returns a Tensor of size size filled with 0. same dtype
            print((C51_U + self.offset).view(-1).max())
            print((C51_U + self.offset).view(-1).min())
            print(target_distribution.view(-1).shape)
            target_distribution.view(-1).index_add_(0, (C51_L + self.offset).view(-1), (next_distribution * (C51_U.float() - C51_b)).view(-1))
            target_distribution.view(-1).index_add_(0, (C51_U + self.offset).view(-1), (next_distribution * (C51_b - C51_L.float())).view(-1))
        return F.binary_cross_entropy_with_logits(input_distribution,target_distribution, reduction='none')
'''
def project_distribution(batch_state, batch_action, non_final_next_states, batch_reward, non_final_mask):
    """
    This is for orignal C51, with KL-divergence.
    """

    with torch.no_grad():
        max_next_dist = torch.zeros((BATCH_SIZE, 1, C51_atoms), device=device, dtype=torch.float)
        max_next_dist += 1.0 / C51_atoms
        #
        max_next_action               = get_action_argmax_next_Q_sa(non_final_next_states)
        if USE_NOISY_NET:
            target_net.sample_noise()
        max_next_dist[non_final_mask] = target_net(non_final_next_states).gather(1, max_next_action)
        max_next_dist = max_next_dist.squeeze()
        #
        # Mapping
        Tz = batch_reward.view(-1, 1) + (GAMMA**NUM_LOOKAHEAD) * C51_support.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
        Tz = Tz.clamp(C51_vmin, C51_vmax)
        C51_b = (Tz - C51_vmin) / C51_delta
        C51_L = C51_b.floor().to(torch.int64)
        C51_U = C51_b.ceil().to(torch.int64)
        C51_L[ (C51_U > 0)               * (C51_L == C51_U)] -= 1
        C51_U[ (C51_L < (C51_atoms - 1)) * (C51_L == C51_U)] += 1
        offset = torch.linspace(0, (BATCH_SIZE - 1) * C51_atoms, BATCH_SIZE)
        offset = offset.unsqueeze(dim=1) 
        offset = offset.expand(BATCH_SIZE, C51_atoms).to(batch_action) # I believe this is to(device)

        # I believe this is analogous to torch.zeros(), but "new_zeros" keeps the type as the original tensor?
        m = batch_state.new_zeros(BATCH_SIZE, C51_atoms) # Returns a Tensor of size size filled with 0. same dtype
        m.view(-1).index_add_(0, (C51_L + offset).view(-1), (max_next_dist * (C51_U.float() - C51_b)).view(-1))
        m.view(-1).index_add_(0, (C51_U + offset).view(-1), (max_next_dist * (C51_b - C51_L.float())).view(-1))
    return m
'''

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