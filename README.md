# 🧠 SAM-for-Semantic-Segmentation

本项目基于 Meta AI 提出的 Segment Anything Model (SAM)，设计并实现了一个 **结构性微调方案**，以适配 **PASCAL VOC 2012** 多类别语义分割任务。

我们在保留 SAM 主干网络结构的前提下，重构其输出掩码模块，实现从“候选前景掩码”到“语义类别分割图”的转变，最终可输出 `[B, 21, H, W]` 维度的语义预测结果。

📌 **项目亮点**：

- ✅ 支持 SAM 结构性微调，仅更新解码器头部
- ✅ 支持 VOC 2012 语义分割任务训练与推理
- ✅ 支持灰度和彩色两种推理输出格式
- ✅ 模块化结构，支持快速迁移与扩展

------

## 📦 项目地址

> 🔗 https://github.com/EmptyCityJK/SAM_for_Semantic_Segmentation/tree/main

---

## 🧰 环境安装

推荐使用 Python 3.8+，并在虚拟环境中安装依赖：

```bash
# 克隆项目
git clone https://github.com/EmptyCityJK/SAM_for_Semantic_Segmentation.git
cd SAM_for_Semantic_Segmentation

# 创建虚拟环境并激活
python -m venv venv
source venv/bin/activate  # Windows 用户用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
````

---

## 🚀 如何运行本项目

### ✅ Step 1：准备数据集

将 **VOC 2012** 数据集解压到如下路径：

```
./DataSet/VOCdevkit/VOC2012/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/
```

> 若需自动下载 VOC 数据，可在网上获取 VOC2012.tar 或使用脚本自动处理。

---

### ✅ Step 2：修改配置文件（可选）

打开 `config/semantic_seg.yaml`，根据需求可配置：

* 模型类型（目前支持 `vit_b`）
* 学习率与冻结策略
* 数据路径与 batch size 等

---

### ✅ Step 3：开始训练

执行以下命令开始训练：

```bash
python train.py
```

训练过程将自动记录：

* 模型参数日志（保存在 `./experiment/model/`）
* TensorBoard 可视化信息（保存在 `./experiment/tensorboard/`）
* WandB 可视化（如启用）

---

### ✅ Step 4：模型推理

训练完成后，可使用以下脚本进行推理：

```bash
# 输出灰度图
python infer.py

# 输出彩色语义图（VOC 色板）
python infer_color.py
```

推理结果将保存在 `./outputs/` 文件夹中。

---

## 🧠 模型结构与微调策略简介

本项目基于原始 SAM 的 `ViT-B` 主干模型，采用如下结构性微调方式：

* ✅ 冻结图像编码器（Image Encoder）
* ✅ 冻结提示编码器（Prompt Encoder），提示全部设为 None
* ✅ 解冻掩码解码器头部的 `output_hypernetworks_mlps` 层，实现对 21 类语义输出建模
* ✅ 每类一个独立 MLP，使用共享 Mask Token 预测像素级类别概率

最终模型输出为形如 `[B, 21, H, W]` 的张量，对每个像素进行 softmax 后即可获得语义预测类别。

---

## 📁 项目结构

```bash
SAM_for_Semantic_Segmentation-main/
├── config/                   # 训练参数配置
├── datasets/                # 数据集加载与预处理
├── extend_sam/              # SAM 模型结构扩展与适配
│   └── segment_anything_ori/ # 原始 SAM 模块（部分重用）
├── losses/                  # 损失函数模块
├── train.py                 # 主训练入口
├── infer.py                 # 推理脚本（灰度图）
├── infer_color.py           # 推理脚本（彩色图）
├── requirements.txt         # 环境依赖列表
├── README.md                # 项目说明文档
└── wandb/                   # wandb 自动生成日志（可选）
```

---

## 📊 可视化与评估

* 📈 使用交叉熵损失与 mIoU 作为主要指标
* 🧮 使用 argmax + softmax 获取像素分类结果
* 📉 WandB 实时记录 loss/mIoU 曲线
* 📷 支持保存灰度/彩色可视化图像用于人工评估

---

## 🤝 引用与致谢

* [Segment Anything (SAM)](https://segment-anything.com/)
* [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
* 感谢 OpenAI 提供 ChatGPT 帮助整理文档结构与实现说明

---

## 📮 联系作者

如有问题或建议，欢迎通过 GitHub Issue 提出，或邮件联系：`emptycityjk@gmail.com`

---

> Star 是对开源最大的鼓励 ⭐，欢迎 Fork 与扩展！
