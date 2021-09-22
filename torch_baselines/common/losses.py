import torch

def weighted_mse_loss(input, target, weight):
    td_errors = input - target
    return (weight * td_errors ** 2).mean(), td_errors.squeeze().detach().numpy()