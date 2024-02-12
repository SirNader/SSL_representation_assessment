import torch.nn.functional as F

from losses.mocov3_loss import mocov3_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase


class Mocov3Head(ContrastiveHeadBase):
    def __init__(self, proj_hidden_dim, pred_hidden_dim, temperature, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.temperature = temperature
        self.projector, self.predictor = None, None
        super().__init__(**kwargs)

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)
        self.predictor = self.create_predictor(output_dim, self.pred_hidden_dim)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        predicted = self.predictor(projected)
        return dict(projected=projected, predicted=predicted)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]
        predicted0 = outputs["view0"]["predicted"]
        predicted1 = outputs["view1"]["predicted"]

        loss0 = mocov3_loss_fn(predicted=predicted1, projected=projected0, temperature=self.temperature)
        loss1 = mocov3_loss_fn(predicted=predicted0, projected=projected1, temperature=self.temperature)
        loss = (loss0 + loss1) / 2

        # calcualte nn accuracy
        normed_projected0 = F.normalize(projected0, dim=-1)
        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)
        return loss, dict(nn_accuracy=nn_acc)
