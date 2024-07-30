# `.\yolov8\ultralytics\models\yolo\pose\val.py`

```py
# 导入必要的模块和类
from pathlib import Path  # 提供处理路径的类
import numpy as np  # 提供数值计算支持
import torch  # 提供深度学习框架支持

# 导入 Ultralytics 内部模块和函数
from ultralytics.models.yolo.detect import DetectionValidator  # 导入检测模型验证器
from ultralytics.utils import LOGGER, ops  # 导入日志和操作函数
from ultralytics.utils.checks import check_requirements  # 导入检查依赖的函数
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou  # 导入评估指标相关函数和常量
from ultralytics.utils.plotting import output_to_target, plot_images  # 导入输出和绘图函数

class PoseValidator(DetectionValidator):
    """
    一个用于基于姿势模型进行验证的 DetectionValidator 的子类。

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml')
        validator = PoseValidator(args=args)
        validator()
        ```py
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化 PoseValidator 对象，设置自定义参数并分配属性。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None  # 初始化 sigma 参数为 None
        self.kpt_shape = None  # 初始化关键点形状参数为 None
        self.args.task = "pose"  # 设置任务类型为 "pose"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)  # 初始化评估指标
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )  # 如果设备为 Apple MPS，发出警告建议使用 CPU 运行姿势模型

    def preprocess(self, batch):
        """预处理批次数据，将 'keypoints' 数据转换为浮点数并移动到指定设备上。"""
        batch = super().preprocess(batch)  # 调用父类方法预处理批次数据
        batch["keypoints"] = batch["keypoints"].to(self.device).float()  # 将关键点数据转换为浮点数并移到设备上
        return batch

    def get_desc(self):
        """返回评估指标的描述信息，以字符串格式返回。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )  # 返回评估指标的表头描述字符串

    def postprocess(self, preds):
        """应用非极大值抑制，返回高置信度得分的检测结果。"""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )  # 对预测结果应用非极大值抑制，返回处理后的检测结果
    # 初始化 YOLO 模型的姿态估计指标
    def init_metrics(self, model):
        # 调用父类的初始化方法，初始化模型指标
        super().init_metrics(model)
        # 获取关键点形状信息
        self.kpt_shape = self.data["kpt_shape"]
        # 判断是否为姿态估计（关键点形状为 [17, 3]）
        is_pose = self.kpt_shape == [17, 3]
        # 关键点数量
        nkpt = self.kpt_shape[0]
        # 根据是否为姿态估计设置高斯核大小
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        # 初始化统计信息字典
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    # 准备批次数据，将关键点转换为浮点数并移到设备上
    def _prepare_batch(self, si, batch):
        # 调用父类的准备批次方法，获取处理后的批次数据
        pbatch = super()._prepare_batch(si, batch)
        # 获取当前批次中的关键点
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        # 获取图像高度和宽度
        h, w = pbatch["imgsz"]
        # 克隆关键点数据
        kpts = kpts.clone()
        # 缩放关键点坐标到图像尺寸
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        # 使用图像尺寸和原始形状比例对关键点坐标进行缩放
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        # 将处理后的关键点数据存入批次中
        pbatch["kpts"] = kpts
        # 返回处理后的批次数据
        return pbatch

    # 准备预测数据，为姿态处理准备并缩放批次中的关键点
    def _prepare_pred(self, pred, pbatch):
        # 调用父类的准备预测方法，获取处理后的预测数据
        predn = super()._prepare_pred(pred, pbatch)
        # 获取关键点数量
        nk = pbatch["kpts"].shape[1]
        # 提取预测关键点坐标并重塑其形状
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        # 使用图像尺寸和原始形状比例对预测关键点坐标进行缩放
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        # 返回处理后的预测数据及预测关键点坐标
        return predn, pred_kpts
    # 定义一个方法用于更新指标数据，接受预测结果和批处理数据作为输入
    def update_metrics(self, preds, batch):
        """Metrics."""
        # 遍历预测结果列表
        for si, pred in enumerate(preds):
            # 增加已处理样本计数器
            self.seen += 1
            # 获取当前预测结果中预测的数量
            npr = len(pred)
            # 初始化统计信息字典，包括置信度、预测类别和真阳性标志
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # 置信度初始化为空张量
                pred_cls=torch.zeros(0, device=self.device),  # 预测类别初始化为空张量
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 真阳性标志初始化为零张量
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 预测关键点真阳性标志初始化为零张量
            )
            # 准备当前批次的数据
            pbatch = self._prepare_batch(si, batch)
            # 分别提取类别和边界框数据
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            # 获取类别数量
            nl = len(cls)
            # 将真实类别和独特的类别 ID 存储到统计信息中
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            
            # 如果预测数量为零，则跳过当前循环
            if npr == 0:
                if nl:
                    # 将统计信息存储到统计数据字典中
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # 如果设置了绘图选项，则处理混淆矩阵
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 如果设置了单类别模式，则将预测结果中的类别分数清零
            if self.args.single_cls:
                pred[:, 5] = 0

            # 准备预测数据和关键点数据
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            # 将预测结果中的置信度和类别分数存储到统计信息中
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 如果存在真实类别，则评估真阳性标志和预测关键点真阳性标志
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"])
                # 如果设置了绘图选项，则处理混淆矩阵
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            # 将当前统计信息存储到统计数据字典中
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 如果设置了保存 JSON 选项，则将预测结果保存为 JSON 格式
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            
            # 如果设置了保存文本文件选项，则将预测结果保存为文本文件
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            detections (torch.Tensor): Tensor with shape (N, 6) representing detection boxes and scores, where each
                detection is of the format (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing ground truth bounding boxes, where each
                box is of the format (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground truth class indices.
            pred_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing predicted keypoints, where
                51 corresponds to 17 keypoints each having 3 values.
            gt_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing ground truth keypoints.

        Returns:
            torch.Tensor: A tensor with shape (N, 10) representing the correct prediction matrix for 10 IoU levels,
                where N is the number of detections.

        Example:
            ```python
            detections = torch.rand(100, 6)  # 100 predictions: (x1, y1, x2, y2, conf, class)
            gt_bboxes = torch.rand(50, 4)  # 50 ground truth boxes: (x1, y1, x2, y2)
            gt_cls = torch.randint(0, 2, (50,))  # 50 ground truth class indices
            pred_kpts = torch.rand(100, 51)  # 100 predicted keypoints
            gt_kpts = torch.rand(50, 51)  # 50 ground truth keypoints
            correct_preds = _process_batch(detections, gt_bboxes, gt_cls, pred_kpts, gt_kpts)
            ```py

        Note:
            `0.53` scale factor used in area computation is referenced from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        if pred_kpts is not None and gt_kpts is not None:
            # 计算每个 ground truth 边界框的面积，乘以 0.53 作为尺度因子
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            # 计算预测关键点与 ground truth 关键点之间的 IoU
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            # 如果没有关键点信息，则计算边界框之间的 IoU
            iou = box_iou(gt_bboxes, detections[:, :4])

        # 返回匹配预测结果的矩阵
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],  # 图像数据
            batch["batch_idx"],  # 批次索引
            batch["cls"].squeeze(-1),  # 类别标签
            batch["bboxes"],  # 预测的边界框
            kpts=batch["keypoints"],  # 预测的关键点
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存文件名
            names=self.names,  # 类别名称
            on_plot=self.on_plot,  # 绘图回调函数
        )
    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        # Concatenate keypoints predictions from all batches of predictions
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        # Plot images with predictions overlaid
        plot_images(
            batch["img"],  # Input images batch
            *output_to_target(preds, max_det=self.args.max_det),  # Convert predictions to target format
            kpts=pred_kpts,  # Predicted keypoints
            paths=batch["im_file"],  # File paths of input images
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # Output filename for the plotted image
            names=self.names,  # List of class names
            on_plot=self.on_plot,  # Callback function for additional plotting actions
        )  # pred

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results
        
        # Save YOLO detections as a TXT file
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # Placeholder image data
            path=None,  # Path not used for saving TXT
            names=self.names,  # List of class names
            boxes=predn[:, :6],  # Detected bounding boxes
            keypoints=pred_kpts,  # Detected keypoints
        ).save_txt(file, save_conf=save_conf)  # Save detections to the specified file

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem  # Get the stem (base filename without extension)
        image_id = int(stem) if stem.isnumeric() else stem  # Convert stem to integer if numeric, else keep as string
        box = ops.xyxy2xywh(predn[:, :4])  # Convert bounding boxes from xyxy to xywh format
        box[:, :2] -= box[:, 2:] / 2  # Adjust bounding box coordinates from xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):  # Iterate over each prediction and adjusted bounding box
            self.jdict.append(  # Append to JSON dictionary
                {
                    "image_id": image_id,  # Image identifier
                    "category_id": self.class_map[int(p[5])],  # Category ID from class map
                    "bbox": [round(x, 3) for x in b],  # Rounded bounding box coordinates
                    "keypoints": p[6:],  # Predicted keypoints
                    "score": round(p[4], 5),  # Confidence score rounded to 5 decimal places
                }
            )
    def eval_json(self, stats):
        """Evaluates object detection model using COCO JSON format."""
        # 检查是否需要保存 JSON，并且确保是 COCO 格式，并且 jdict 不为空
        if self.args.save_json and self.is_coco and len(self.jdict):
            # 设置注释文件和预测文件的路径
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            # 打印评估信息，包括预测文件和注释文件的路径
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                # 检查是否满足 pycocotools 的要求
                check_requirements("pycocotools>=2.0.6")
                # 导入 pycocotools 的相关模块
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                # 确保注释文件和预测文件存在
                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                # 初始化 COCO 对象，用于处理注释
                anno = COCO(str(anno_json))  # init annotations api
                # 加载预测结果到 COCO 对象中，必须传入字符串而不是 Path 对象
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                # 遍历两种评估（bbox 和 keypoints）
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
                    # 如果是 COCO 数据集，设置要评估的图片 IDs
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    # 执行评估
                    eval.evaluate()
                    # 累积评估结果
                    eval.accumulate()
                    # 汇总评估指标
                    eval.summarize()
                    # 更新 mAP50-95 和 mAP50 的统计数据到 stats 字典中
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                # 捕获异常，打印警告信息
                LOGGER.warning(f"pycocotools unable to run: {e}")
        # 返回更新后的 stats 字典
        return stats
```