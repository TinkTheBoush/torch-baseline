import torch

def weighted_mse_loss(input, target, weight):
    td_errors = input - target
    return (weight * td_errors ** 2).mean(), td_errors.squeeze().detach().numpy()


def categorial_loss(input_distribution,next_distribution,categorial_bar,next_categorial_bar):
    bar_start = categorial_bar[:-1]
    bar_end = categorial_bar[1:]
    target_distribution = next_distribution
    x = target_distribution - input_distribution
    loss = -(x*torch.log(x)).mean(1).mean(0)
    return loss