import torch
import torch.nn.functional as F

from losses.simclr_loss import simclr_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase


class SimclrHead(ContrastiveHeadBase):
    def __init__(self, temperature, proj_hidden_dim, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.projector = None
        super().__init__(**kwargs)
        self.temperature = temperature

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        return dict(projected=projected)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]
        projected = torch.cat([projected0, projected1])

        # calculate loss
        n_views = len([None for output_key in outputs.keys() if "view" in output_key])
        repeated_idx = idx.repeat(n_views)
        loss = simclr_loss_fn(projected, indexes=repeated_idx, temperature=self.temperature)

        # calcualte nn accuracy
        normed_projected0 = F.normalize(projected0, dim=-1)
        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)

        return loss, dict(nn_accuracy=nn_acc)
