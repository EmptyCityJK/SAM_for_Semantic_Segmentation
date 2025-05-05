import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .segment_anything_ori.modeling.common import LayerNorm2d


class OriHead(nn.Module):

    def __init__(
            self,
            *,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs + 1
        # 特征图上采样
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        # 超网络, 每个 mask token 都对应一个 MLP，用于将其转换为和上采样后的特征图进行点积生成 mask
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        # 用于生成 IOU 质量评分的 MLP，输出维度是 num_mask_tokens
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            src: torch.Tensor,
            iou_token_out: torch.Tensor,
            mask_tokens_out: torch.Tensor,
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
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        b, c, h, w = src.shape

        # Upscale mask embeddings and predict masks using the mask tokens
        # 转换维度形状为卷积所需的 [B, C, H, W] 格式
        src = src.transpose(1, 2).view(b, c, h, w)
        # 将 Transformer 输出的低分辨率特征图上采样
        upscaled_embedding = self.output_upscaling(src)
        # 对每个 mask token 通过 MLP 生成一个权重向量，用于预测对应的 mask
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) # hyper_in: [B, class_num, C]
        
        b, c, h, w = upscaled_embedding.shape
        # 每类 mask 的生成：类动态特征 * 上采样图像特征
        # 将 hyper_in 与 upscaled_embedding 做矩阵乘法预测 mask，最终恢复 [B, C, H, W]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions（生成 IOU 分数）
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1) # 只输出第一个最精确的
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        # masks shape: [batch_size, num_multimask_outputs, 256, 256]
        return masks, iou_pred


class SemSegHead(nn.Module):

    def __init__(
            self,
            *,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            class_num: int = 20,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.class_num = class_num

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        # 每个类别一个超网络MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.class_num)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            src: torch.Tensor,
            iou_token_out: torch.Tensor,
            mask_tokens_out: torch.Tensor,
            src_shape,
            mask_scale=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          src (torch.Tensor): The tensor contains image embedding and sparse prompt embedding
          iou_token_out (torch.Tensor): Tokens of iou prediction from neck module
          mask_tokens_out (torch.Tensor): Tokens of mask prediction form neck module
          mask_scale (int): Original SAM output 3 masks which is from local to global as default
                            This Class use one of three mask tokens to transform it into class-ware
                            semantic segmentation prediction

        Returns:
          torch.Tensor: batched predicted semantic masks
          torch.Tensor: batched predictions of mask quality
        """
        b, c, h, w = src_shape

        # 上采样特征图Upscale mask embeddings and predict masks using the mask tokens
        # src: [B, 64*64, 256] → transpose(1, 2) → [B, 256, 4096] → reshape → [B, 256, 64, 64]
        src = src.transpose(1, 2).view(b, c, h, w)
        # upscaled: ConvTranspose2d (x2 x2) → [B, 64, 256, 256]
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        # 每个类别都用一个 MLP 对同一个 mask token（通过 mask_scale 指定）处理
        for i in range(self.class_num):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, mask_scale, :])) # mask_tokens_out[:, mask_scale, :] shape [B, 256]
        # [B, num_classes, 32]
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        # 生成 [B, class_num, H, W] 的语义分割 mask
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # B N H W, N is num of category

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # B N H W, N is num of category

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
