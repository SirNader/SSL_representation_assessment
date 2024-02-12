import torch.nn.functional as F

from losses.vicreg_loss import vicreg_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase


class VicregHead(ContrastiveHeadBase):
    def __init__(self, sim_loss_weight, var_loss_weight, cov_loss_weight, proj_hidden_dim, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.projector = None
        super().__init__(**kwargs)
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim, last_batchnorm=False)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        return dict(projected=projected)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]

        loss = vicreg_loss_fn(
            projected0,
            projected1,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        # calcualte nn accuracy
        normed_projected0 = F.normalize(projected0, dim=-1)
        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)
        return loss, dict(nn_accuracy=nn_acc)
