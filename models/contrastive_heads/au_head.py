import torch.nn.functional as F

from distributed.gather import all_gather_grad
from loggers.functional.alignment import alignment_loss
from loggers.functional.uniformity import uniformity_loss
from .base.contrastive_head_base import ContrastiveHeadBase


class AuHead(ContrastiveHeadBase):
    def __init__(self, proj_hidden_dim, alignment_loss_weight, uniformity_loss_weight, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.projector = None
        super().__init__(**kwargs)
        self.alignment_loss_weight = alignment_loss_weight
        self.uniformity_loss_weight = uniformity_loss_weight

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        return dict(projected=projected)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]

        # calculate losses
        gathered_projected0 = all_gather_grad(projected0)
        gathered_projected1 = all_gather_grad(projected1)
        loss1 = alignment_loss(gathered_projected0, gathered_projected1)
        loss2 = uniformity_loss(gathered_projected0)
        loss3 = uniformity_loss(gathered_projected1)
        loss4 = (loss2 + loss3) / 2
        loss = loss1 * self.alignment_loss_weight + loss4 * self.uniformity_loss_weight

        # calcualte nn accuracy
        normed_projected0 = F.normalize(projected0, dim=-1)
        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)

        return loss, dict(nn_accuracy=nn_acc)
