import torch
from extend_sam.extend_sam import BaseExtendSam, SemanticSam
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import cv2

# 定义变换列表
transform = Compose([
    Resize(size=(1024, 1024)),  # 将图像resize到256x256
    ToTensor()  # 将PIL图像或NumPyndarray转换为Tensor，并归一化至[0,1]
])

model_path = "experiment/model/semantic_sam/model.pth"
sam_seg = SemanticSam(class_num=21)

sam_seg.load_state_dict(torch.load(model_path), strict=False)

imgpath = 'DataSet/VOCdevkit/VOC2012/JPEGImages/2007_000733.jpg'
img = Image.open(imgpath).convert('RGB')
width, height = img.size
image = transform(img)
print(image.shape)

image = torch.unsqueeze(image, dim=0)
masks_pred, iou_pred = sam_seg(image)
print(masks_pred.shape)
masks_pred = F.interpolate(masks_pred, (height, width), mode="bicubic", align_corners=False)
print(masks_pred.shape)
output_softmax = F.softmax(masks_pred, dim=1)
output_argmax = torch.argmax(output_softmax, dim=1, keepdim=True)
print(output_argmax.shape)
print(np.unique(output_argmax.detach().numpy()))
res = output_argmax.detach().numpy()[0, 0, :, :]
res = res * 10 + 50

res3Chanel = cv2.merge([res, res, res])
cv2.imwrite("mask.png", res3Chanel)
print(res3Chanel.shape)