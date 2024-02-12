import torch.nn as nn

from .projector_head import ProjectorHead


class ProjectorPredictorHead(ProjectorHead):
    def __init__(self, pred_hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.pred_hidden_dim = pred_hidden_dim
        self.predictor = self.create_predictor()

    def create_predictor(self):
        # TODO check exact architectures per method
        # this is the predictor according to NNCLR paper
        return nn.Sequential(
            nn.Linear(self.output_dim, self.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(self.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pred_hidden_dim, self.output_dim, bias=False),
            # nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        output = super().forward(x)
        output["predicted"] = self.predictor(output["projected"])
        return output
