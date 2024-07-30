# `.\yolov8\ultralytics\models\yolo\segment\val.py`

```py
# 导入所需模块
from multiprocessing.pool import ThreadPool
from pathlib import Path

# 导入 NumPy 和 PyTorch 库
import numpy as np
import torch
import torch.nn.functional as F

# 导入 Ultralytics 相关模块和函数
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images

# 定义一个继承自 DetectionValidator 的 SegmentationValidator 类
class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```py
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        # 调用父类的初始化方法
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        # 初始化额外的属性
        self.plot_masks = None
        self.process = None
        # 将任务设置为 'segment'，并初始化评估指标为 SegmentMetrics
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        # 调用父类的预处理方法
        batch = super().preprocess(batch)
        # 将批次中的 masks 转换为 float 类型，并发送到设备上
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        # 调用父类的初始化评估指标方法
        super().init_metrics(model)
        # 初始化绘制 masks 的列表
        self.plot_masks = []
        # 如果设置了保存为 JSON 格式，则检查所需的 pycocotools 版本
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # 根据保存标志选择处理 masks 的函数
        # 如果设置了保存为 JSON 或 TXT，则选择更精确的本地处理函数
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        # 初始化统计信息字典
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        # 返回格式化的评估指标描述字符串
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )
    def postprocess(self, preds):
        """
        Post-processes YOLO predictions and returns output detections with proto.

        Args:
            preds (list): List of prediction outputs from YOLO model.

        Returns:
            tuple: A tuple containing processed predictions (p) and prototype data (proto).
        """
        # Perform non-maximum suppression on the first prediction output
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        # Determine the prototype data from the second prediction output
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto

    def _prepare_batch(self, si, batch):
        """
        Prepares a batch for training or inference by processing images and targets.

        Args:
            si (int): Index of the current sample in the batch.
            batch (dict): Dictionary containing batch data including images and targets.

        Returns:
            dict: A prepared batch dictionary with additional 'masks' data.
        """
        # Call superclass method to prepare the batch
        prepared_batch = super()._prepare_batch(si, batch)
        # Determine which indices to use for masks based on overlap_mask flag
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        # Add masks data to the prepared batch
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """
        Prepares predictions for training or inference by processing images and targets.

        Args:
            pred (Tensor): Predictions from the model.
            pbatch (dict): Prepared batch data.
            proto (Tensor): Prototype data for processing masks.

        Returns:
            tuple: A tuple containing processed predictions (predn) and processed masks (pred_masks).
        """
        # Call superclass method to prepare predictions
        predn = super()._prepare_pred(pred, pbatch)
        # Process masks using prototype data and prediction outputs
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks
    # 更新评估指标的方法
    def update_metrics(self, preds, batch):
        """Metrics."""
        # 遍历预测结果的每个样本
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            # 增加已处理样本计数
            self.seen += 1
            # 计算当前预测的数量
            npr = len(pred)
            # 初始化统计数据结构
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # 置信度列表
                pred_cls=torch.zeros(0, device=self.device),  # 预测类别列表
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # True Positive 列表
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # True Positive for Masked 列表
            )
            # 准备批次数据
            pbatch = self._prepare_batch(si, batch)
            # 分离出类别和边界框数据
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            # 计算目标类别和独特类别
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            
            # 如果没有预测结果，但有真实标签
            if npr == 0:
                if nl:
                    # 将统计数据添加到总体统计中
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # 如果需要绘图，处理混淆矩阵
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 处理掩膜数据
            gt_masks = pbatch.pop("masks")
            
            # 准备预测数据
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 如果有真实标签，评估预测结果
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                # 如果需要绘图，处理混淆矩阵
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            # 将统计数据添加到总体统计中
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 转换预测掩膜为Tensor，并添加到绘图列表中
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # 选取前15个样本进行绘图

            # 保存预测结果到JSON文件
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            # 保存预测结果到文本文件
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
    def finalize_metrics(self, *args, **kwargs):
        """
        Sets speed and confusion matrix for evaluation metrics.
        """
        # 将速度和混淆矩阵设置为评估指标中的属性值
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Example:
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```py
        """
        if masks:
            # 如果处理的是带有掩码数据的批次
            if overlap:
                # 如果要考虑重叠的掩码
                nl = len(gt_cls)
                # 创建索引并扩展掩码以匹配预测掩码的形状
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                # 如果地面真实掩码的形状与预测掩码的形状不匹配，进行插值操作
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            # 计算掩码的 IoU
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # 处理框
            # 计算框的 IoU
            iou = box_iou(gt_bboxes, detections[:, :4])

        # 返回匹配预测结果
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        # 使用自定义函数 plot_images 绘制验证样本图像，并添加边界框标签
        plot_images(
            batch["img"],  # 图像数据
            batch["batch_idx"],  # 批次索引
            batch["cls"].squeeze(-1),  # 压缩类别信息
            batch["bboxes"],  # 边界框信息
            masks=batch["masks"],  # 可选参数，掩膜信息
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存文件名
            names=self.names,  # 类别名称映射
            on_plot=self.on_plot,  # 绘图回调函数
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        # 使用自定义函数 plot_images 绘制预测结果图像，包括掩膜和边界框
        plot_images(
            batch["img"],  # 图像数据
            *output_to_target(preds[0], max_det=15),  # 将预测转换为目标格式，最多15个检测结果
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,  # 组合绘制的掩膜信息
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存文件名
            names=self.names,  # 类别名称映射
            on_plot=self.on_plot,  # 绘图回调函数
        )  # pred
        self.plot_masks.clear()  # 清空掩膜列表

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        # 使用 Results 类保存 YOLO 检测结果到文本文件，使用指定的格式和坐标
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建一个全零数组作为占位符
            path=None,  # 不保存路径信息
            names=self.names,  # 类别名称映射
            boxes=predn[:, :6],  # 边界框信息
            masks=pred_masks,  # 掩膜信息
        ).save_txt(file, save_conf=save_conf)  # 调用 Results 类的 save_txt 方法保存文本文件

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        # 导入 pycocotools.mask 中的 encode 函数
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            # 将预测的掩膜编码为 RLE，并追加到结果字典 jdict 中
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")  # 将编码后的 counts 字段解码为 UTF-8 格式
            return rle

        stem = Path(filename).stem  # 获取文件名的主干部分
        image_id = int(stem) if stem.isnumeric() else stem  # 如果主干部分是数字，则转换为整数作为 image_id
        box = ops.xyxy2xywh(predn[:, :4])  # 将边界框格式从 xyxy 转换为 xywh
        box[:, :2] -= box[:, 2:] / 2  # 将边界框的中心点坐标转换为左上角坐标
        pred_masks = np.transpose(pred_masks, (2, 0, 1))  # 转置掩膜数据的维度顺序
        with ThreadPool(NUM_THREADS) as pool:  # 使用线程池并行处理
            rles = pool.map(single_encode, pred_masks)  # 并行编码掩膜数据
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):  # 遍历预测结果和边界框
            self.jdict.append(  # 将结果以字典形式追加到 jdict 中
                {
                    "image_id": image_id,  # 图像 ID
                    "category_id": self.class_map[int(p[5])],  # 类别 ID，通过 class_map 映射获取
                    "bbox": [round(x, 3) for x in b],  # 边界框坐标，保留三位小数
                    "score": round(p[4], 5),  # 分数，保留五位小数
                    "segmentation": rles[i],  # 掩膜编码结果
                }
            )
    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        # 检查是否需要保存 JSON，并且数据格式为 COCO，并且 jdict 不为空
        if self.args.save_json and self.is_coco and len(self.jdict):
            # 定义标注文件和预测文件的路径
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            # 记录评估过程中使用的文件
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                # 检查并导入 pycocotools 所需的版本
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                # 确保注释文件和预测文件存在
                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                # 初始化 COCO 对象用于注释
                anno = COCO(str(anno_json))  # init annotations api
                # 加载预测结果用于 COCO 对象
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                # 进行两种评估：bbox 和 segm
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    # 如果是 COCO 格式，设置图像 IDs 用于评估
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    # 更新统计信息中的 mAP50-95 和 mAP50
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                # 捕获异常并记录警告信息
                LOGGER.warning(f"pycocotools unable to run: {e}")
        # 返回更新后的统计信息
        return stats
```