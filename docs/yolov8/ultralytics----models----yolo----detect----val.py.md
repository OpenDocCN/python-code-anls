# `.\yolov8\ultralytics\models\yolo\detect\val.py`

```py
# 导入所需的库和模块
import os
from pathlib import Path

import numpy as np
import torch

# 导入自定义的数据处理模块
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
# 导入自定义的验证器基类
from ultralytics.engine.validator import BaseValidator
# 导入自定义的工具模块
from ultralytics.utils import LOGGER, ops
# 导入检查要求的函数
from ultralytics.utils.checks import check_requirements
# 导入评估指标相关模块
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
# 导入绘图函数
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```py
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        # 调用父类构造函数初始化基本变量和设置
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        # 初始化特定于检测模型的变量
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"  # 设置任务为检测任务
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)  # 初始化评估指标
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU向量，用于计算mAP@0.5:0.95
        self.niou = self.iouv.numel()  # 计算IoU向量的长度
        self.lb = []  # 用于自动标注的列表
        if self.args.save_hybrid:
            # 如果设置了保存混合结果，发出警告，说明可能会影响mAP的正确性
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        # 将图像批处理移到设备上，并根据需要进行半精度转换和归一化
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        # 将其他信息也移到设备上
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            # 如果设置了保存混合结果
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            # 调整边界框坐标
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            # 为自动标注构建标签列表
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch
    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # 获取验证数据集路径
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # 判断是否为 COCO 数据集
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # 判断是否为 LVIS 数据集
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))  # 根据数据集类型选择类别映射
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # 如果是 COCO 或 LVIS 数据集且非训练阶段，设置保存 JSON 结果
        self.names = model.names  # 获取模型的类别名称列表
        self.nc = len(model.names)  # 类别数量
        self.metrics.names = self.names  # 设置评估指标的类别名称
        self.metrics.plot = self.args.plots  # 设置是否绘制图像
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)  # 初始化混淆矩阵
        self.seen = 0  # 初始化 seen 参数
        self.jdict = []  # 初始化 jdict 列表
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])  # 初始化统计信息字典

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")  # 返回格式化的描述字符串，总结 YOLO 模型的类别指标

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )  # 对预测输出应用非最大抑制算法，返回处理后的预测结果

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si  # 确定当前批次中与索引 si 对应的样本
        cls = batch["cls"][idx].squeeze(-1)  # 获取类别信息并去除多余的维度
        bbox = batch["bboxes"][idx]  # 获取边界框信息
        ori_shape = batch["ori_shape"][si]  # 获取原始图像形状
        imgsz = batch["img"].shape[2:]  # 获取图像尺寸
        ratio_pad = batch["ratio_pad"][si]  # 获取比例填充信息
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # 转换目标框坐标格式并缩放
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # 缩放边界框到原始空间
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}  # 返回处理后的验证批次数据

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()  # 克隆预测结果
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # 缩放预测框到原始空间
        return predn  # 返回处理后的预测结果
    def update_metrics(self, preds, batch):
        """
        Update metrics based on predictions and batch data.

        Args:
            preds (list): List of predictions.
            batch (dict): Batch data dictionary.

        Returns:
            None
        """
        # Iterate over predictions
        for si, pred in enumerate(preds):
            self.seen += 1  # Increment the count of processed predictions
            npr = len(pred)  # Number of predictions in the current batch item

            # Initialize statistics dictionary
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # Confidence values tensor
                pred_cls=torch.zeros(0, device=self.device),  # Predicted classes tensor
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # True positives tensor
            )

            # Prepare the batch for processing
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")  # Extract class labels and bounding boxes
            nl = len(cls)  # Number of ground truth labels in the batch

            # Set target class labels and unique image-level classes
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            # Skip further processing if there are no predictions
            if npr == 0:
                if nl:
                    # Append statistics to respective lists
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # Optionally process confusion matrix if plots are enabled
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Adjust predictions if single class mode is enabled
            if self.args.single_cls:
                pred[:, 5] = 0

            # Prepare predictions for processing
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]  # Confidence values from predictions
            stat["pred_cls"] = predn[:, 5]  # Predicted classes from predictions

            # Evaluate predictions if ground truth labels are present
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)  # Calculate true positives
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)  # Update confusion matrix

            # Append statistics to respective lists
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save predictions in JSON format if enabled
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])

            # Save predictions in text file format if enabled
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metric values after all predictions are processed.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        # Set final speed metric and confusion matrix values
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """
        Retrieve metrics statistics and results.

        Returns:
            dict: Dictionary containing metrics statistics.
        """
        # Convert statistics tensors to numpy arrays
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        
        # Calculate number of ground truth labels per class and per image
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        
        # Remove target_img key from statistics dictionary
        stats.pop("target_img", None)
        
        # Process metrics if there are any true positive predictions
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        
        # Return metrics results dictionary
        return self.metrics.results_dict
    def print_results(self):
        """
        Prints training/validation set metrics per class.
        """
        # 设置打印格式，包括所有类别的计数、每类的样本数、各指标的均值结果
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        # 使用日志记录器打印所有类别的汇总信息
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        # 如果数据集中没有标签，警告用户无法计算指标
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # 按类别打印结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            # 对每个类别打印详细的指标结果
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        # 如果设置了绘图选项，则绘制混淆矩阵
        if self.args.plots:
            # 分别绘制归一化和非归一化的混淆矩阵
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        # 计算检测结果与真实边界框的 IoU
        iou = box_iou(gt_bboxes, detections[:, :4])
        # 返回匹配预测结果的矩阵，用于不同 IoU 水平的评估
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 构建 YOLO 数据集
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.
        """
        # 构建数据集
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        # 构建并返回数据加载器
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        # 调用plot_images函数，绘制验证集图像样本，保存为指定文件
        plot_images(
            batch["img"],  # 图像数据
            batch["batch_idx"],  # 批次索引
            batch["cls"].squeeze(-1),  # 类别标签
            batch["bboxes"],  # 边界框信息
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存的文件名
            names=self.names,  # 类别名称列表
            on_plot=self.on_plot,  # 绘图回调函数
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # 调用plot_images函数，绘制输入图像上的预测边界框，并保存结果
        plot_images(
            batch["img"],  # 图像数据
            *output_to_target(preds, max_det=self.args.max_det),  # 将预测结果转换为目标格式
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存的文件名
            names=self.names,  # 类别名称列表
            on_plot=self.on_plot,  # 绘图回调函数
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        # 创建Results对象，保存YOLO检测结果到指定的txt文件中，使用归一化的坐标
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建空白图像作为占位
            path=None,  # 文件路径（暂未指定）
            names=self.names,  # 类别名称列表
            boxes=predn[:, :6],  # 检测框信息
        ).save_txt(file, save_conf=save_conf)  # 调用Results对象的保存方法

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem  # 获取文件名的主干部分
        image_id = int(stem) if stem.isnumeric() else stem  # 解析图像ID
        box = ops.xyxy2xywh(predn[:, :4])  # 将边界框从xyxy格式转换为xywh格式
        box[:, :2] -= box[:, 2:] / 2  # 将xy中心坐标转换为左上角坐标
        for p, b in zip(predn.tolist(), box.tolist()):  # 遍历每个预测和其对应的边界框
            self.jdict.append(  # 将结果添加到JSON字典中
                {
                    "image_id": image_id,  # 图像ID
                    "category_id": self.class_map[int(p[5])] + (1 if self.is_lvis else 0),  # 类别ID
                    "bbox": [round(x, 3) for x in b],  # 边界框坐标
                    "score": round(p[4], 5),  # 检测置信度得分
                }
            )
    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""

        # 检查是否需要保存 JSON，并且数据集为 COCO 或 LVIS，且 jdict 非空
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            # 设置预测结果保存路径
            pred_json = self.save_dir / "predictions.json"  # predictions
            # 设置注释文件路径
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations

            # 根据数据集类型确定使用的包
            pkg = "pycocotools" if self.is_coco else "lvis"

            # 打印评估信息
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")

            try:
                # 检查预测结果和注释文件是否存在
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"

                # 检查依赖的版本要求
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")

                if self.is_coco:
                    # COCO 数据集的评估过程
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    # 初始化 COCO 数据集的注释 API 和预测 API
                    anno = COCO(str(anno_json))
                    pred = anno.loadRes(str(pred_json))
                    val = COCOeval(anno, pred, "bbox")
                else:
                    # LVIS 数据集的评估过程
                    from lvis import LVIS, LVISEval

                    # 初始化 LVIS 数据集的注释 API 和预测 API
                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    val = LVISEval(anno, pred, "bbox")

                # 设置需要评估的图像 ID 列表
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                # 执行评估过程
                val.evaluate()
                # 累积评估结果
                val.accumulate()
                # 总结评估结果
                val.summarize()

                if self.is_lvis:
                    # 如果是 LVIS 数据集，显示详细评估结果
                    val.print_results()

                # 更新统计指标 mAP50-95 和 mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )

            except Exception as e:
                # 捕获异常并记录警告信息
                LOGGER.warning(f"{pkg} unable to run: {e}")

        # 返回更新后的统计指标字典
        return stats
```