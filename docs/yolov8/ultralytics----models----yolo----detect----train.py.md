# `.\yolov8\ultralytics\models\yolo\detect\train.py`

```py
# 导入必要的库和模块
import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

# 导入 Ultralytics 的相关模块和函数
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

# 定义一个名为 DetectionTrainer 的类，继承自 BaseTrainer 类，用于检测模型的训练
class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```py
    """

    # 定义 build_dataset 方法，用于构建 YOLO 数据集
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 获取模型的最大步长
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # 调用 build_yolo_dataset 函数构建 YOLO 数据集
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    # 定义 get_dataloader 方法，用于构建和返回数据加载器
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        # 确保 mode 参数合法
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        # 如果使用分布式训练（DDP），使用 torch_distributed_zero_first 函数初始化数据集 *.cache 以确保只初始化一次
        with torch_distributed_zero_first(rank):
            # 调用 build_dataset 方法构建数据集
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        # 根据 mode 确定是否需要打乱数据集
        shuffle = mode == "train"
        # 如果 dataset 具有 rect 属性且需要打乱，则警告并设置 shuffle=False
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        # 根据 mode 确定 workers 数量
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        # 调用 build_dataloader 函数构建数据加载器并返回
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        # 将图像批次移到指定设备上，并转换为浮点数，同时进行归一化（除以255）
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        # 如果启用了多尺度处理
        if self.args.multi_scale:
            imgs = batch["img"]
            # 随机生成一个介于指定范围内的尺度大小
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            # 计算尺度因子
            sf = sz / max(imgs.shape[2:])  # scale factor
            # 如果尺度因子不为1，则进行插值操作，调整图像尺寸
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            # 更新批次中的图像数据
            batch["img"] = imgs
        # 返回预处理后的批次数据
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # 以下三行被注释掉的代码是关于模型属性设置的尝试，可能是用于调整超参数的缩放
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        # 将类别数量（nc）、类别名称和超参数附加到模型对象上
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        # 创建一个 YOLO 检测模型实例
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        # 如果提供了预训练权重，则加载到模型中
        if weights:
            model.load(weights)
        # 返回创建的模型实例
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        # 设置损失名称列表
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        # 返回一个用于 YOLO 模型验证的 DetectionValidator 实例
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        # 构建包含带有标签的训练损失项的字典
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        # 如果提供了损失项，则将其转换为五位小数的浮点数，返回标签化的损失字典
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            # 如果没有提供损失项，返回损失项名称列表
            return keys
    # 返回一个格式化的训练进度字符串，包括 epoch、GPU 内存、损失、实例数和大小等信息
    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    # 绘制带有注释的训练样本图像
    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    # 绘制来自 CSV 文件的指标图表
    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    # 创建一个带标签的 YOLO 模型训练图
    def plot_training_labels(self):
        # 从训练数据集的标签中获取所有边界框并连接起来
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        # 获取所有类别并连接起来
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        # 绘制带标签的训练图，使用数据集的类别名称和保存目录
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
```