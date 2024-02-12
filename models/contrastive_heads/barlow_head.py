import torch.nn.functional as F

from losses.barlow_loss import barlow_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase


class BarlowHead(ContrastiveHeadBase):
    def __init__(self, off_diagonal_scale, proj_hidden_dim, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.projector = None
        super().__init__(**kwargs)
        self.off_diagonal_scale = off_diagonal_scale

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim, last_batchnorm=False)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        return dict(projected=projected)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]

        loss = barlow_loss_fn(projected0, projected1, off_diagonal_scale=self.off_diagonal_scale)

        # calcualte nn accuracy
        normed_projected0 = F.normalize(projected0, dim=-1)
        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)
        return loss, dict(nn_accuracy=nn_acc)
