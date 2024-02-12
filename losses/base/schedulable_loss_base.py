import torch.nn as nn


class SchedulableLossBase(nn.Module):
    def __init__(self, update_counter):
        super().__init__()
        self.update_counter = update_counter
