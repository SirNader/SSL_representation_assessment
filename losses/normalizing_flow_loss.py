import torch.nn as nn


class NormalizingFlowLoss(nn.Module):
    @staticmethod
    def forward(z, log_det_jac, reduction="batch"):
        """
        reduction == "batch" --> return scalar
        reduction == "sample" --> return a scalar for each sample
        reduction == "dimension" --> return a scalar for each dimension
        """
        assert reduction in ["batch", "sample", "dimension"]
        # likelihood for normal distribution with mean=0 std=1
        # 1 / sqrt(2 * pi) * e ^ (-0.5 * x ^ 2)
        # log likelihood = ln(1 / sqrt(2 * pi)) - 0.5 * x ^ 2
        # LL = ln(1) - ln(sqrt(2pi) - 0.5x²
        # NLL = 0.5x²
        flat_z = z.flatten(start_dim=1)
        # note normalization is done for the whole term as the flat_z term and log_det_jac
        # are both a sum with flat_z.shape[1] summands
        # return (0.5 * (flat_z ** 2).sum(dim=1) - log_det_jac).mean() / flat_z.shape[1]
        z_loss = 0.5 * (flat_z ** 2)
        jac_loss = -log_det_jac
        if reduction == "batch":
            z_loss, jac_loss = NormalizingFlowLoss._reduce_batch(z_loss, jac_loss)
        elif reduction == "sample":
            z_loss, jac_loss = NormalizingFlowLoss._reduce_sample(z_loss, jac_loss)
        return z_loss, jac_loss / flat_z.shape[1]

    @staticmethod
    def _reduce_batch(z_loss, jac_loss):
        # sum dimensions for each sample
        n_dims = z_loss.shape[1]
        z_loss = z_loss.sum(dim=1)
        # average over samples
        z_loss = z_loss.mean()
        jac_loss = jac_loss.mean()
        # z_loss was summed --> normalize
        return z_loss / n_dims, jac_loss

    @staticmethod
    def _reduce_sample(z_loss, jac_loss):
        # sum dimensions for each sample
        n_dims = z_loss.shape[1]
        z_loss = z_loss.sum(dim=1)
        # z_loss was summed --> normalize
        return z_loss / n_dims, jac_loss
