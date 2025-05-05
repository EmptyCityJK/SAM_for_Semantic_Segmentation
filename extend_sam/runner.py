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
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        writer = None
        if cfg.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f"{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/")
        
        best_valid_mIoU = -1
        model_path = f"{cfg.model_folder}/{cfg.experiment_name}/model.pth"
        log_path = f"{cfg.log_folder}/{cfg.experiment_name}/log_file.txt"
        check_folder(model_path)
        check_folder(log_path)

        for epoch in range(cfg.max_epoch):
            # ------------------- 训练 -------------------
            self.model.train()
            train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
            train_metric = mIoUOnline(self.train_loader.dataset.class_names)

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

                predictions = torch.argmax(masks_pred, dim=1)
                for i in range(images.size(0)):
                    pred_mask = get_numpy_from_tensor(predictions[i])
                    gt_mask = get_numpy_from_tensor(labels[i])
                    gt_mask = cv2.resize(gt_mask, pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    train_metric.add(pred_mask, gt_mask)

                loss_dict['total_loss'] = total_loss.item()
                train_meter.add(loss_dict)

            self.scheduler.step()

            train_loss = train_meter.get(clear=True)['total_loss']
            train_miou, train_miou_fg = train_metric.get(clear=True)

            # ------------------- val mIoU 和 loss -------------------
            (val_miou, val_miou_fg), val_loss = self._eval_with_loss(cfg)

            # ------------------- Save best model -------------------
            if val_miou > best_valid_mIoU:
                best_valid_mIoU = val_miou
                save_model(self.model, model_path, parallel=self.the_number_of_gpu > 1)
                print_and_save_log(f"Epoch {epoch+1}: saved best model to {model_path}", path=log_path)

            # ------------------- Logging -------------------
            log_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_mIoU": train_miou,
                "train_mIoU_fg": train_miou_fg,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "val_mIoU_fg": val_miou_fg,
                "val_best_mIoU": best_valid_mIoU
            }
            write_log(epoch + 1, log_path, log_data, status="epoch", writer=writer, timer=self.train_timer)
            # wandb 日志记录
            wandb.log(log_data)

        save_model(self.model, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)
        if writer:
            writer.close()


    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                masks_pred, iou_pred = self.model(images)
                predictions = torch.argmax(masks_pred, dim=1)
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index].squeeze(0))
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric.add(pred_mask, gt_mask)
        self.model.train()
        return eval_metric.get(clear=True)

    def _eval_with_loss(self, cfg):
        # 用于获取验证 loss 和 mIoU
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
                    gt_mask = cv2.resize(gt_mask, pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    eval_metric.add(pred_mask, gt_mask)

        self.model.train()
        return eval_metric.get(clear=True), val_meter.get(clear=True)['total_loss']

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels
            if loss_cfg[item[0]].label_one_hot:
                class_num = cfg.model.params.class_num
                real_labels = one_hot_embedding_3d(real_labels, class_num=class_num) ###  b h w --> b class_num h w
            tmp_loss = item[1](mask_pred, real_labels) ## cross entropy loss ,mask_pred 大小也为 b class_num h w
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]].weight * tmp_loss
