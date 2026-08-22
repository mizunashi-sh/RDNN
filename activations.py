import torch

def rectified_tanh(x):
    return torch.max(torch.zeros_like(x), torch.tanh(x))

def softplus(x):
    return torch.log1p(torch.exp(x))
