# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from .segment_anything_ori.modeling.common import LayerNorm2d

'''
This file save the mask_decoder's neck class, 
which is the former part of original mask decoder of SAM. 
Then the mask_decoder_heads can be used with the neck.
'''


class MaskDecoderNeck(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3, # 同时输出的 mask 数量，默认为 3
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        # 用于输出 mask 的质量（IoU）
        self.iou_token = nn.Embedding(1, transformer_dim) # iou shape [1, 256]
        self.num_mask_tokens = num_multimask_outputs + 1
        # 多个 token 用于生成多个不同粒度的 mask（local/global...）
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim) # mask shape [4, 256]
        # 特征上采样模块
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: The tensor contains image embedding and sparse prompt embedding(transformer 后的图像特征)
          torch.Tensor: Tokens of iou prediction
          torch.Tensor: Tokens of mask prediction
        """
        # Concatenate output tokens（Token 构建与拼接）
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens shape: [batch_size, T, 256] T = 1(IOU) + 4(MASK) + N(prompt)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask（图像特征扩展与加位置编码）
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        src_shape = src.shape # [batch_size, 256, 64, 64]
        # Run the transformer（Transformer 编码）
        # hs [batch_size, T, 256] 
        # src [batch_size, 64*64, 256]
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token shape: [batch_size, 1, 256]
        iou_token_out = hs[:, 0, :]
        # mask_token shape: [batch_size, num_mask_tokens, 256]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        return src, iou_token_out, mask_tokens_out, src_shape
