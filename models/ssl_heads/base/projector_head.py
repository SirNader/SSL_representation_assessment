import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed.gather import all_gather_nograd
from models.base.single_model_base import SingleModelBase
from models.poolings.single_pooling import SinglePooling
from utils.factory import create


class ProjectorHead(SingleModelBase):
    def __init__(
            self,
            proj_hidden_dim,
            output_dim,
            queue_size=None,
            pooling=None,
            detach=False,
            output_shape=None,
            backbone=None,
            **kwargs,
    ):
        assert output_shape is None
        super().__init__(output_shape=(output_dim,), **kwargs)
        self.detach = detach
        self.pooling = create(pooling, SinglePooling, model=backbone) or nn.Identity()
        input_shape = self.pooling(torch.ones(1, *self.input_shape)).shape[1:]

        self.input_dim = np.prod(input_shape)
        self.proj_hidden_dim = proj_hidden_dim
        self.output_dim = output_dim

        self.projector = self.create_projector()

        # use queue independent of method as an online evaluation metric
        self.queue_size = queue_size
        if queue_size is not None:
            self.register_buffer("queue", F.normalize(torch.randn(self.queue_size, output_dim), dim=1))
            self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("queue_id", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def last_batchnorm(self):
        return True

    def create_projector(self):
        # TODO check exact architectures per method
        # this is the projector according to NNCLR paper
        layers = [
            nn.Linear(self.input_dim, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.output_dim, bias=False),
        ]
        if self.last_batchnorm:
            # some methods use affine=False here
            # layers.append(nn.BatchNorm1d(output_dim, affine=False))
            layers.append(nn.BatchNorm1d(self.output_dim))
        return nn.Sequential(*layers)

    def _prepare_forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        if self.detach:
            x = x.detach()
        x = self.pooling(x).flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self._prepare_forward(x)
        return dict(projected=self.projector(x))

    @torch.no_grad()
    def calculate_nn_accuracy(self, normed_projected0, y, ids, idx0=None, nn0=None):
        if idx0 is None and nn0 is None:
            # nnclr already found idx0 and nn0
            idx0, nn0 = self.find_nn(normed_projected0, ids=ids)
        nn_acc = ((y == self.queue_y[idx0]).sum() / len(y)).item()
        self.dequeue_and_enqueue(normed_projected0, y=y, ids=ids)
        return nn_acc

    @torch.no_grad()
    def get_queue_similarity_matrix(self, normed_projected, ids):
        similarity_matrix = normed_projected @ self.queue.T
        # exclude the same sample of the previous epoch
        is_own_id = self.queue_id[None, :] == ids[:, None]
        # set similarity to self to -1
        similarity_matrix[is_own_id] = -1.
        return similarity_matrix

    @torch.no_grad()
    def find_nn(self, normed_projected, ids):
        similarity_matrix = self.get_queue_similarity_matrix(normed_projected, ids=ids)
        idx = similarity_matrix.max(dim=1)[1]
        nearest_neighbor = self.queue[idx]
        return idx, nearest_neighbor

    @torch.no_grad()
    def dequeue_and_enqueue(self, normed_projected0, y, ids):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            normed_projected0 (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
            ids (torch.Tensor): ids of the samples in the batch.
        """
        # disable in eval mode (for automatic batch_size finding)
        if not self.training:
            return

        normed_projected0 = all_gather_nograd(normed_projected0)
        y = all_gather_nograd(y)
        ids = all_gather_nograd(ids)

        batch_size = normed_projected0.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.queue[ptr:ptr + batch_size] = normed_projected0
        self.queue_y[ptr:ptr + batch_size] = y
        self.queue_id[ptr:ptr + batch_size] = ids
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
