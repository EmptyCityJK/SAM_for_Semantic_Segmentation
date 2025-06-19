import torch
from extend_sam.extend_sam import SemanticSam
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import cv2

# VOC调色板
VOC_COLORMAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128),
]

# 图像变换
transform = Compose([
    Resize(size=(1024, 1024)),
    ToTensor()
])

# 模型加载
model_path = "experiment/model/semantic_sam/model.pth"
sam_seg = SemanticSam(class_num=21)
sam_seg.load_state_dict(torch.load(model_path), strict=False)
sam_seg.eval()

# 输入图像
imgpath = 'DataSet/VOCdevkit/VOC2012/JPEGImages/2007_000733.jpg'
img = Image.open(imgpath).convert('RGB')
width, height = img.size
image = transform(img).unsqueeze(0)

# 推理
with torch.no_grad():
    masks_pred, _ = sam_seg(image)
    masks_pred = F.interpolate(masks_pred, size=(height, width), mode="bilinear", align_corners=False)
    output = torch.argmax(F.softmax(masks_pred, dim=1), dim=1).squeeze().cpu().numpy()

# 伪彩色输出
color_mask = np.zeros((height, width, 3), dtype=np.uint8)
for cls_id, color in enumerate(VOC_COLORMAP):
    color_mask[output == cls_id] = color

# 保存结果
cv2.imwrite("mask_color.png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
