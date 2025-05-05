from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
import wandb


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)
        print(f"Using {self.the_number_of_gpu} GPUs: {use_gpu}")


class SemRunner(BaseRunner):
    def train(self, cfg):        
        best_val_mIoU = -1
        model_path = f"{cfg.model_folder}/{cfg.experiment_name}/model.pth"

        for epoch in range(cfg.max_epoch):
            self.model.train()
            train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
            train_metric = mIoUOnline(self.train_loader.dataset.class_names)

            print("Start training loop: Epoch {}".format(epoch + 1))
            for images, labels in self.train_loader:
                images, labels = images.cuda(), labels.cuda().long()
                masks_pred, _ = self.model(images)
                masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)

                total_loss = torch.zeros(1).cuda()
                loss_dict = {}
                self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # 对每个像素在类别维度 class_num 上取最大值，得到概率最高的类别索引
                predictions = torch.argmax(masks_pred, dim=1)
                for i in range(images.size(0)):
                    pred_mask = get_numpy_from_tensor(predictions[i])
                    gt_mask = get_numpy_from_tensor(labels[i])
                    train_metric.add(pred_mask, gt_mask)

                loss_dict['total_loss'] = total_loss.item()
                train_meter.add(loss_dict)

            self.scheduler.step()

            train_loss = train_meter.get(clear=True)['total_loss']
            train_mIoU, train_mIoU_fg = train_metric.get(clear=True)
            (val_mIoU, val_mIoU_fg), val_loss = self._eval_with_loss(cfg)

            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                save_model(self.model, model_path, parallel=self.the_number_of_gpu > 1)
                print(f"Epoch {epoch+1}: saved best model to {model_path}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_mIoU": train_mIoU,
                "train_mIoU_fg": train_mIoU_fg,
                "val_loss": val_loss,
                "val_mIoU": val_mIoU,
                "val_mIoU_fg": val_mIoU_fg,
                "val_best_mIoU": best_val_mIoU
            })

        save_model(self.model, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)

    def _eval_with_loss(self, cfg):
        self.model.eval()
        class_names = self.val_loader.dataset.class_names
        val_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        eval_metric = mIoUOnline(class_names)

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.cuda(), labels.cuda().long()
                masks_pred, _ = self.model(images)
                masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)

                total_loss = torch.zeros(1).cuda()
                loss_dict = {}
                self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)
                val_meter.add({**loss_dict, 'total_loss': total_loss.item()})

                predictions = torch.argmax(masks_pred, dim=1)
                for i in range(images.size(0)):
                    pred_mask = get_numpy_from_tensor(predictions[i])
                    gt_mask = get_numpy_from_tensor(labels[i])
                    eval_metric.add(pred_mask, gt_mask)

        self.model.train()
        return eval_metric.get(clear=True), val_meter.get(clear=True)['total_loss']

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        for loss_name, loss_fn in self.losses.items():
            real_labels = labels
            if cfg.losses[loss_name].label_one_hot:
                class_num = cfg.model.params.class_num
                real_labels = one_hot_embedding_3d(real_labels, class_num=class_num)
            tmp_loss = loss_fn(mask_pred, real_labels)
            loss_dict[loss_name] = tmp_loss.item()
            total_loss += cfg.losses[loss_name].weight * tmp_loss
