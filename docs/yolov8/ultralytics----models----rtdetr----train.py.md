# `.\yolov8\ultralytics\models\rtdetr\train.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入所需模块和库
from copy import copy
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr
from .val import RTDETRDataset, RTDETRValidator

# 定义 RT-DETRTrainer 类，继承自 DetectionTrainer 类
class RTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection. Extends the DetectionTrainer
    class for YOLO to adapt to the specific features and architecture of RT-DETR. This model leverages Vision
    Transformers and has capabilities like IoU-aware query selection and adaptable inference speed.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```py
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model='rtdetr-l.yaml', data='coco8.yaml', imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```py
    """

    # 获取模型方法，初始化并返回用于对象检测任务的 RT-DETR 模型
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration. Defaults to None.
            weights (str, optional): Path to pre-trained model weights. Defaults to None.
            verbose (bool): Verbose logging if True. Defaults to True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    # 构建数据集方法，返回用于训练或验证的 RT-DETR 数据集对象
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training. Defaults to None.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    # 获取验证器方法，返回适用于 RT-DETR 模型验证的 DetectionValidator 对象
    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    # 继承父类方法，预处理图像批次。将图像缩放并转换为浮点格式。
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        # 调用父类的预处理方法，获取预处理后的批次数据
        batch = super().preprocess_batch(batch)
        
        # 获取批次中图像的数量
        bs = len(batch["img"])
        
        # 获取当前批次的索引
        batch_idx = batch["batch_idx"]
        
        # 初始化用于存储真实边界框和类别的列表
        gt_bbox, gt_class = [], []
        
        # 遍历批次中的每张图像
        for i in range(bs):
            # 将当前批次索引等于 i 的边界框添加到 gt_bbox 中，并将其移到相应设备上
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            
            # 将当前批次索引等于 i 的类别添加到 gt_class 中，并将其移到相应设备上
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        
        # 返回预处理后的批次数据
        return batch
```