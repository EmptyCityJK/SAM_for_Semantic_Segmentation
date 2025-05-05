import torch
import torch.nn as nn
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter


class BaseExtendSam(nn.Module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        # 从注册器中加载对应结构的原始 SAM 模型(sam_b)，并加载预训练权重
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        # 是否冻结图像编码器
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        # 是否冻结提示编码器
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        # 是否冻结掩码解码器（False）
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)

    def forward(self, img):
        # img shape: [batch_size, 3, 1024, 1024]
        # 得到图像特征 x(图像嵌入)，即 SAM 的图像编码器输出
        x = self.img_adapter(img) # x shape: [batch_size, 256, 64, 64]
        # 未启用提示输入
        points = None
        boxes = None
        masks = None
        # 获取稀疏（points, boxes）和密集（mask）编码，用于解码器输入
        # 稀疏编码 shape: [batch_size, 256, 1, 1]
        # 密集编码 shape: [batch_size, 256, 64, 64]
        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        ) 
        # 是否输出多种掩码（SAM 的默认特性，语义分割可能只取一个）
        multimask_output = True
        # 使用解码器生成低分辨率掩码及预测的 IOU
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x, # 图像嵌入特征
            prompt_adapter=self.prompt_adapter, # 提示编码器
            sparse_embeddings=sparse_embeddings, # 稀疏编码
            dense_embeddings=dense_embeddings, # 密集编码
            multimask_output=multimask_output, # 是否输出多种掩码
        )
        # low_res_masks shape: [batch_size, 3, 256, 256]
        return low_res_masks, iou_predictions

# 为语义分割任务专门封装的类
class SemanticSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, class_num=20, model_type='vit_b'):
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        # 输出为多类别分割图
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)
