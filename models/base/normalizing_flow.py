import torch

from losses.normalizing_flow_loss import NormalizingFlowLoss
from .single_model_base import SingleModelBase


class NormalizingFlow(SingleModelBase):
    def forward(self, x):
        raise NotImplementedError

    def backward(self, z):
        raise NotImplementedError

    def predict_binary(self, x):
        """ returns a score per sample """
        scores = {}
        z, log_det = self(x)

        # NLL (loss is equal to NLL)
        z_loss, jac_loss = NormalizingFlowLoss.forward(z, log_det, reduction="sample")
        scores["mean_NLL"] = z_loss + jac_loss
        z_loss, jac_loss = NormalizingFlowLoss.forward(z, log_det, reduction="dimension")
        scores["max_NLL"] = z_loss.max(dim=1)[0] + jac_loss

        # z^2: DifferNet/CSFlow square z as score (https://github.com/marco-rudolph/cs-flow/issues/18)
        flat_z = z.flatten(start_dim=1)
        z_squared = flat_z.pow(2)
        scores["mean_z^2"] = z_squared.mean(dim=1)
        scores["max_z^2"] = z_squared.max(dim=1)[0]
        return scores

    def sample(self, batch_size):
        z = torch.randn(batch_size, *self.input_shape)
        samples, _ = self.backward(z)
        return samples
