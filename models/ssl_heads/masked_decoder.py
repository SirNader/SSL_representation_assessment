from functools import partial

import einops
import torch
import torch.nn as nn

from initializers.functional import initialize_linear_bias_to_zero, initialize_layernorm_as_noaffine
from models.modules.vit_block import VitBlock
from utils.param_checking import to_2tuple, float_to_integer_exact
from utils.positional_embedding import get_2d_sincos_pos_embed, interpolate_pos_embed_temporary
from utils.vit_util import get_sequence_lengths
from ..base.single_model_base import SingleModelBase


class MaskedDecoder(SingleModelBase):
    def __init__(self, embedding_dim, depth, attention_heads, backbone, pred_patch_size=None, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = to_2tuple(backbone.patch_size)
        self.pred_patch_size = self.patch_size if pred_patch_size is None else to_2tuple(pred_patch_size)
        self.n_aux_tokens = backbone.n_aux_tokens
        self.output_shape = backbone.input_shape

        self.embedding_dim = embedding_dim
        self.depth = depth
        self.attention_heads = attention_heads

        # decoder doesn't produce original image shape but flattened patches
        flat_patch_dim = self.pred_patch_size[0] * self.pred_patch_size[1] * self.output_shape[0]
        num_tokens, encoder_embedding_dim = self.input_shape
        # remember original output_shape (e.g. for unpatchify)
        self.original_output_shape = self.output_shape
        self.output_shape = (num_tokens, flat_patch_dim)

        self.embed = nn.Linear(encoder_embedding_dim, embedding_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        # fixed embedding
        self.register_buffer("pos_embed", torch.zeros(1, num_tokens - self.n_aux_tokens, embedding_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=embedding_dim,
                num_heads=attention_heads,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embedding_dim)
        # decoder to patch
        self.pred = nn.Linear(embedding_dim, flat_patch_dim, bias=True)

    def load_state_dict(self, state_dict, strict=True):
        old_pos_embed = state_dict["pos_embed"]
        # LEGACY: old checkpoints have pos_embed that stores zeros for cls token
        old_n_positions = old_pos_embed.shape[1]
        if int(old_n_positions ** 0.5) ** 2 + 1 == old_n_positions:
            state_dict["pos_embed"] = old_pos_embed[:, 1:]
        # LEGACY: end
        super().load_state_dict(state_dict=state_dict, strict=strict)

    @property
    def _requires_initializer(self):
        return False

    def _model_specific_initialization(self):
        # mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize pos_embed with sin-cos embedding
        _, img_height, img_width = self.original_output_shape
        h_seqlen, w_seqlen = get_sequence_lengths(self.patch_size, img_height, img_width)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.embedding_dim, h_seqlen, w_seqlen)
        self.pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.apply(initialize_layernorm_as_noaffine)
        self.apply(initialize_linear_bias_to_zero)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)

    # noinspection PyMethodOverriding
    def forward(self, inputs):
        x = inputs["x"]
        ids_restore = inputs["ids_restore"]

        # embed tokens
        x = self.embed(x)

        # extract shapes
        bs, n_input_tokens, dim = x.shape
        _, total_n_patches = ids_restore.shape
        n_hidden_patches = total_n_patches - (n_input_tokens - self.n_aux_tokens)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(bs, n_hidden_patches, 1)
        # no aux tokens
        all_patches = torch.cat([x[:, self.n_aux_tokens:, :], mask_tokens], dim=1)
        # unshuffle
        indices = einops.repeat(ids_restore, "bs np -> bs np dim", dim=dim)
        all_patches = torch.gather(all_patches, dim=1, index=indices)

        # add pos embed
        if total_n_patches != self.pos_embed.shape[1]:
            _, old_h, old_w = self.original_output_shape
            old_token_h = int(old_h / self.patch_size[0])
            old_token_w = int(old_w / self.patch_size[1])
            # assume image was scaled the same in all directions (relevant for non-square images)
            scale = (self.pos_embed.shape[1] / total_n_patches) ** 0.5
            new_token_h = float_to_integer_exact(old_token_h / scale)
            new_token_w = float_to_integer_exact(old_token_w / scale)
            pos_embed = interpolate_pos_embed_temporary(
                old_pos_embed=self.pos_embed,
                old_token_h=old_token_h,
                old_token_w=old_token_w,
                new_token_h=new_token_h,
                new_token_w=new_token_w,
            )
        else:
            pos_embed = self.pos_embed
        all_patches = all_patches + pos_embed

        # append aux tokens
        x = torch.cat([x[:, :self.n_aux_tokens, :], all_patches], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        # remove aux token
        x = x[:, self.n_aux_tokens:, :]

        return x
