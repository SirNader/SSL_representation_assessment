import torch.nn as nn


# derive from nn.Module to be usable in nn.Sequential
class PoolingBase(nn.Module):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)
