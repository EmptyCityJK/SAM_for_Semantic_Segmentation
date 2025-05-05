
import torch.nn as nn
import torch
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
from .segment_anything_ori.modeling.mask_decoder import MaskDecoder
from typing import List, Tuple
from torch.nn import functional as F
from .mask_decoder_heads import SemSegHead
from .mask_decoder_neck import MaskDecoderNeck


class BaseMaskDecoderAdapter(MaskDecoder):
    '''
      multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    '''

    # is fix and load params
    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseMaskDecoderAdapter, self).__init__(transformer_dim=ori_sam.mask_decoder.transformer_dim,
                                                     transformer=ori_sam.mask_decoder.transformer)
        self.sam_mask_decoder = ori_sam.mask_decoder
        if fix: # 冻结参数
            fix_params(self.sam_mask_decoder)  # move to runner to implement

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(image_embeddings=image_embeddings,
                                                                   # prompt encoder 的位置编码
                                                                   image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                   sparse_prompt_embeddings=sparse_embeddings,
                                                                   dense_prompt_embeddings=dense_embeddings,
                                                                   multimask_output=multimask_output, )
        return low_res_masks, iou_predictions

# 用于多类别语义分割任务，替换原始 mask decoder 的输出结构
class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False, class_num=20):
        super(SemMaskDecoderAdapter, self).__init__(ori_sam, fix)
        # 中间特征生成模块，用于生成 mask token、iou token
        self.decoder_neck = MaskDecoderNeck(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                            transformer=self.sam_mask_decoder.transformer,
                                            num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs)
        # 输出语义分割图的头部模块。负责将 mask token 映射为语义 mask
        self.decoder_head = SemSegHead(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                       num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs,
                                       iou_head_depth=self.sam_mask_decoder.iou_head_depth,
                                       iou_head_hidden_dim=self.sam_mask_decoder.iou_head_hidden_dim,
                                       class_num=class_num)
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        # 使用 pair_params 将原始 mask decoder 的权重拷贝到新 neck/head 中
        self.pair_params(self.decoder_neck)
        self.pair_params(self.decoder_head)

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True,
                scale=1):
        # 模拟标准分割网络结构，先用 neck 提取中间 token 表征，再用 head 输出多类别掩码
        src, iou_token_out, mask_tokens_out, src_shape = self.decoder_neck(image_embeddings=image_embeddings,
                                                                           image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                                           dense_prompt_embeddings=dense_embeddings,
                                                                           multimask_output=multimask_output, )
        masks, iou_pred = self.decoder_head(src, iou_token_out, mask_tokens_out, src_shape, mask_scale=scale)
        return masks, iou_pred
    # 用于从原始 decoder 中复制参数权重到新定义的模块（neck/head），实现权重迁移
    def pair_params(self, target_model: nn.Module):
        src_dict = self.sam_mask_decoder.state_dict()
        for name, value in target_model.named_parameters():
            if name in src_dict.keys():
                value.data.copy_(src_dict[name].data)


# Lightly adapted from（用于 IOU 预测器）
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
