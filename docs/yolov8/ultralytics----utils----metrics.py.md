# `.\yolov8\ultralytics\utils\metrics.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

# Object Keypoint Similarity (OKS) sigmas for different keypoints
OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area calculation
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area calculation
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # Convert box coordinates to float for accurate computation
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    
    # Calculate intersection area
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # Compute IoU
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    This function calculates IoU considering different variants such as Generalized IoU (GIoU),
    Distance IoU (DIoU), and Complete IoU (CIoU) if specified.

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box of shape (1, 4).
        box2 (torch.Tensor): A tensor representing multiple bounding boxes of shape (n, 4).
        xywh (bool, optional): If True, treats boxes as (x_center, y_center, width, height).
        GIoU (bool, optional): If True, compute Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, compute Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, compute Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU values between box1 and each box in box2, of shape (n,).
    """
    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # 根据输入的格式标志，获取边界框的坐标信息
    if xywh:  # 如果输入格式为 (x, y, w, h)
        # 将 box1 和 box2 按照坐标和尺寸分块
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        # 计算各自的一半宽度和高度
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        # 计算边界框的四个顶点坐标
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # 如果输入格式为 (x1, y1, x2, y2)
        # 将 box1 和 box2 按照坐标分块
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        # 计算边界框的宽度和高度，并添加一个小值 eps 避免除以零
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 计算并集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # 计算 IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        # 计算最小包围框的宽度和高度
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:  # 如果是 Distance IoU 或者 Complete IoU
            # 计算最小包围框的对角线的平方
            c2 = cw.pow(2) + ch.pow(2) + eps
            # 计算中心距离的平方
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4
            if CIoU:  # 如果是 Complete IoU
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # 计算 CIoU
            return iou - rho2 / c2  # 计算 DIoU
        # 计算最小包围框的面积
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area  # 计算 GIoU
    return iou  # 返回 IoU
# 计算两个方向边界框之间的概率 IoU，参考论文 https://arxiv.org/pdf/2106.06072v1.pdf
def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing the first oriented bounding boxes in xywhr format.
        obb2 (torch.Tensor): A tensor of shape (M, 5) representing the second oriented bounding boxes in xywhr format.
        CIoU (bool, optional): If True, compute Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing the probabilities of IoU between obb1 and obb2.
    """
    # 将 Gaussian 边界框合并，忽略中心点（前两列）因为这里不需要
    gbbs = torch.cat((obb1[:, 2:4].pow(2) / 12, obb1[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    # 计算旋转边界框的协方差矩阵
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    # Splitting the x and y coordinates from obb1 and obb2
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)

    # Calculating covariance matrix components for obb1 and obb2
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    # Calculation of terms t1, t2, and t3 for IoU computation
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5

    # Combined term for boundary distance
    bd = (t1 + t2 + t3).clamp(eps, 100.0)

    # Hausdorff distance calculation
    hd = (1.0 - (-bd).exp() + eps).sqrt()

    # Intersection over Union (IoU) computation
    iou = 1 - hd

    # Compute Complete IoU (CIoU) if CIoU flag is True
    if CIoU:
        # Splitting width and height components from obb1 and obb2
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)

        # Calculating v value based on width and height ratios
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)

        # Compute alpha factor and adjust IoU for CIoU
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        return iou - v * alpha  # CIoU

    # Return regular IoU if CIoU flag is False
    return iou
# 计算两个有方向边界框之间的概率IoU，参考论文 https://arxiv.org/pdf/2106.06072v1.pdf
def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    # 将输入obb1和obb2转换为torch.Tensor，如果它们是np.ndarray类型的话
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    # 分割xy坐标和宽高比例与旋转角度信息，以便后续处理
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    
    # 计算相关的协方差矩阵分量
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    # 计算概率IoU的三个部分
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    
    # 组合三个部分，并进行一些修正和限制
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    
    # 返回1减去修正的IoU概率
    return 1 - hd


# 计算平滑的正负二元交叉熵目标
def smooth_BCE(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
    """
    # 计算平滑后的正负二元交叉熵目标
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """
    # 用于计算和更新目标检测和分类任务的混淆矩阵的类定义
    def __init__(self, task, nc, conf=0.5, iou_thres=0.5):
        self.task = task
        self.matrix = np.zeros((nc, nc), dtype=np.int64)
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    # 更新混淆矩阵中的条目
    def update_matrix(self, targets, preds):
        """
        Update the confusion matrix with new target and prediction entries.

        Args:
            targets (np.ndarray): An array containing the ground truth labels.
            preds (np.ndarray): An array containing the predicted labels.
        """
        for t, p in zip(targets, preds):
            self.matrix[t, p] += 1

    # 重置混淆矩阵
    def reset_matrix(self):
        """Reset the confusion matrix to all zeros."""
        self.matrix.fill(0)

    # 打印混淆矩阵的当前状态
    def print_matrix(self):
        """Print the current state of the confusion matrix."""
        print(self.matrix)
    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """
        Initialize attributes for the YOLO model.

        Args:
            nc (int): Number of classes.
            conf (float): Confidence threshold, default is 0.25, adjusted to 0.25 if None or 0.001.
            iou_thres (float): IoU (Intersection over Union) threshold.
            task (str): Task type, either "detect" or other.

        Initializes:
            self.task (str): Task type.
            self.matrix (np.ndarray): Confusion matrix initialized based on task type and number of classes.
            self.nc (int): Number of classes.
            self.conf (float): Confidence threshold.
            self.iou_thres (float): IoU threshold.
        """
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1
    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        # 检查标签是否为空
        if gt_cls.shape[0] == 0:
            if detections is not None:
                # 根据置信度阈值过滤掉低置信度的检测结果
                detections = detections[detections[:, 4] > self.conf]
                # 提取检测结果的类别
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # 假阳性
            return
        
        # 如果没有检测结果
        if detections is None:
            # 提取真实标签的类别
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # 背景 FN
            return

        # 根据置信度阈值过滤掉低置信度的检测结果
        detections = detections[detections[:, 4] > self.conf]
        # 提取真实标签的类别
        gt_classes = gt_cls.int()
        # 提取检测结果的类别
        detection_classes = detections[:, 5].int()
        # 判断是否为带有角度信息的检测结果和真实标签
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5
        # 计算 IoU（交并比）
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        # 根据 IoU 阈值筛选匹配结果
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        # 判断是否有匹配结果
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        # 更新混淆矩阵
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确
            else:
                self.matrix[self.nc, gc] += 1  # 真实背景

        # 如果有匹配结果，更新混淆矩阵
        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # 预测背景

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix
    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # 提取混淆矩阵的对角线元素，即 true positives
        fp = self.matrix.sum(1) - tp  # 计算每行的和减去对角线元素，得到 false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections) -- 该行被注释掉，不起作用
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # 如果任务是检测，移除背景类别后返回结果

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn  # 引入 seaborn 库，用于绘制混淆矩阵图

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # 对混淆矩阵进行列归一化处理
        array[array < 0.005] = np.nan  # 将小于 0.005 的值设为 NaN，不在图上标注

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)  # 创建图和轴对象，设置图的大小和布局
        nc, nn = self.nc, len(names)  # 类别数和类别名称列表的长度
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # 设置字体大小，根据类别数决定
        labels = (0 < nn < 99) and (nn == nc)  # 根据类别名称是否符合要求决定是否应用于刻度标签
        ticklabels = (list(names) + ["background"]) if labels else "auto"  # 根据条件设置刻度标签
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略警告信息，避免空矩阵的 RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,  # 如果类别数小于 30，则在图上标注数值
                annot_kws={"size": 8},  # 标注的字体大小
                cmap="Blues",  # 使用蓝色调色板
                fmt=".2f" if normalize else ".0f",  # 数值格式，归一化时保留两位小数，否则取整数
                square=True,  # 方形图
                vmin=0.0,  # 最小值为 0
                xticklabels=ticklabels,  # X 轴刻度标签
                yticklabels=ticklabels,  # Y 轴刻度标签
            ).set_facecolor((1, 1, 1))  # 设置图的背景色为白色
        title = "Confusion Matrix" + " Normalized" * normalize  # 图表标题，根据是否归一化添加后缀
        ax.set_xlabel("True")  # X 轴标签
        ax.set_ylabel("Predicted")  # Y 轴标签
        ax.set_title(title)  # 设置图表标题
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'  # 图片保存的文件名
        fig.savefig(plot_fname, dpi=250)  # 保存图表为 PNG 文件，设置 DPI 为 250
        plt.close(fig)  # 关闭图表
        if on_plot:
            on_plot(plot_fname)  # 如果有回调函数，则调用该函数，并传递图表文件路径

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):  # 循环打印混淆矩阵的每一行
            LOGGER.info(" ".join(map(str, self.matrix[i])))  # 将每一行转换为字符串并记录到日志中
def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # 计算曲线下面积，使用梯形法则
    ap = np.sum(np.diff(mrec) * mpre[:-1])

    return ap, mpre, mrec
    method = "interp"  # 定义变量 method，并赋值为 "interp"，表示采用插值法计算平均精度
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 在 [0, 1] 区间生成101个均匀间隔的点，用于插值计算 (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # 使用梯形法则计算插值后的曲线下面积，得到平均精度
    else:  # 如果 method 不是 "interp"，则执行 'continuous' 分支
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 找到 mrec 中 recall 值发生变化的索引位置
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 计算曲线下面积，得到平均精度

    return ap, mpre, mrec  # 返回计算得到的平均精度 ap，以及修改后的 mpre 和 mrec
    # 根据对象置信度降序排列索引
    i = np.argsort(-conf)
    
    # 按照排序后的顺序重新排列 tp, conf, pred_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找出唯一的类别和它们的数量
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别的数量，也是检测的数量

    # 创建 Precision-Recall 曲线并计算每个类别的平均精度 (AP)
    x, prec_values = np.linspace(0, 1, 1000), []

    # 初始化存储平均精度 (AP)，精度 (Precision)，和召回率 (Recall) 曲线的数组
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        # ci 是类别 c 在 unique_classes 中的索引，c 是当前类别的值
        i = pred_cls == c
        # 计算预测类别为 c 的样本数
        n_l = nt[ci]  # number of labels
        # 计算真实类别为 c 的样本数
        n_p = i.sum()  # number of predictions
        # 如果没有预测类别为 c 的样本或者真实类别为 c 的样本，则跳过
        if n_p == 0 or n_l == 0:
            continue

        # 累积计算假阳性和真阳性
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 计算召回率
        recall = tpc / (n_l + eps)  # recall curve
        # 在负向 x 上插值，以生成召回率曲线
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # 计算精确率
        precision = tpc / (tpc + fpc)  # precision curve
        # 在负向 x 上插值，以生成精确率曲线
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # 从召回率-精确率曲线计算平均准确率
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            # 如果需要绘图并且是第一个类别，记录在 mAP@0.5 处的精确率值
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # 计算 F1 值（精确率和召回率的调和平均数）
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    # 仅保留有数据的类别名称列表
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # 转换为字典形式
    # 如果需要绘图，则绘制精确率-召回率曲线、F1 曲线、精确率曲线、召回率曲线
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    # 找到最大 F1 值所在的索引
    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    # 获取最大 F1 值对应的精确率、召回率、F1 值
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    # 计算真正例（TP）
    tp = (r * nt).round()  # true positives
    # 计算假正例（FP）
    fp = (tp / (p + eps) - tp).round()  # false positives
    # 返回结果：TP、FP、精确率、召回率、F1 值、平均准确率、唯一类别、精确率曲线、召回率曲线、F1 曲线、x 值、精确率值
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # Precision for each class, initialized as an empty list
        self.r = []  # Recall for each class, initialized as an empty list
        self.f1 = []  # F1 score for each class, initialized as an empty list
        self.all_ap = []  # AP scores for all classes and IoU thresholds, initialized as an empty list
        self.ap_class_index = []  # Index of class for each AP score, initialized as an empty list
        self.nc = 0  # Number of classes, initialized to 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []  # Return AP50 values if all_ap is not empty, otherwise an empty list

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at IoU thresholds from 0.5 to 0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with mean AP values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []  # Return mean AP values across IoU thresholds if all_ap is not empty, otherwise an empty list

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0  # Return the mean precision of classes if p is not empty, otherwise 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0  # Return the mean recall of classes if r is not empty, otherwise 0.0
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0



    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0



    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0



    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]



    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]



    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps



    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()



    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results



    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []



    @property
    def pr_curves(self):
        """Returns precision and recall curves."""
        return self.p_curve, self.r_curve



    @property
    def f1_curves(self):
        """Returns F1 score curves."""
        return self.f1_curve



    @property
    def precision_values(self):
        """Returns precision values for the PR curve."""
        return self.px, self.prec_values



    @property
    def pr_values(self):
        """Returns precision and recall values."""
        return self.p, self.r



    @property
    def f1_values(self):
        """Returns F1 values."""
        return self.f1



    @property
    def pr(self):
        """Returns precision and recall."""
        return self.p, self.r



    @property
    def f1(self):
        """Returns F1 score."""
        return self.f1



    def print_results(self):
        """Prints results (p, r, ap50, ap)."""
        print(self.p, self.r, self.ap50, self.ap)



    def evaluation(self):
        """Model evaluation with metric AP."""
        return self.ap



    def result(self):
        """Return p, r, ap50, ap."""
        return self.p, self.r, self.ap50, self.ap



    @property
    def recall(self):
        """Returns recall."""
        return self.r



    @property
    def mean(self):
        """Returns the mean AP."""
        return self.ap.mean()



    @property
    def mapss(self):
        """Returns mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps



    @property
    def model(self):
        """Returns the model."""
        return self.model
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        返回一个包含多个曲线的列表，用于访问特定的度量曲线。

        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            # 返回包含 Precision 和 Recall 曲线的列表，使用 self.px 作为 x 轴，self.prec_values 作为 y 轴

            [self.px, self.f1_curve, "Confidence", "F1"],
            # 返回包含 F1 曲线的列表，使用 self.px 作为 x 轴，self.f1_curve 作为 y 轴

            [self.px, self.p_curve, "Confidence", "Precision"],
            # 返回包含 Precision 曲线的列表，使用 self.px 作为 x 轴，self.p_curve 作为 y 轴

            [self.px, self.r_curve, "Confidence", "Recall"],
            # 返回包含 Recall 曲线的列表，使用 self.px 作为 x 轴，self.r_curve 作为 y 轴
        ]
class DetMetrics(SimpleClass):
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """
        Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names.
        """
        # 设置保存输出图表的目录路径，默认为当前目录
        self.save_dir = save_dir
        # 是否绘制每个类别的精度-召回率曲线的标志，默认为 False
        self.plot = plot
        # 可选的回调函数，用于在绘制完成时传递图表路径和数据，默认为 None
        self.on_plot = on_plot
        # 类别名称的元组，表示检测模型所涉及的类别名称，默认为空元组
        self.names = names
        # Metric 类的实例，用于存储检测指标的结果
        self.box = Metric()
        # 存储检测过程中不同部分执行时间的字典
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        # 任务类型，这里为检测任务
        self.task = "detect"
    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        # 返回一个包含特定指标键的列表，用于访问特定指标数据
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        # 计算检测到的对象的平均值，并返回精度、召回率、mAP50 和 mAP50-95
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        # 返回评估特定类别对象检测模型性能的结果
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        # 返回每个类别的平均精度 (mAP) 分数
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        # 返回盒子对象的适应性（健壮性）
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        # 返回每个类别的平均精度指数
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        # 返回计算的性能指标和统计数据的字典
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # 返回用于访问特定指标曲线的曲线列表
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        # 返回计算的性能指标和统计数据的字典
        return self.box.curves_results
# SegmentMetrics 类，继承自 SimpleClass，用于计算和聚合给定类别集合上的检测和分割指标。

class SegmentMetrics(SimpleClass):
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, callback function, and class names."""
        # 初始化保存结果图像的目录路径
        self.save_dir = save_dir
        # 是否保存检测和分割图像的标志
        self.plot = plot
        # 可选的回调函数，用于在图像渲染时传递图像路径和数据
        self.on_plot = on_plot
        # 类别名称列表
        self.names = names
        # Metric 类的实例，用于计算盒子检测指标
        self.box = Metric()
        # Metric 类的实例，用于计算分割掩码指标
        self.seg = Metric()
        # 存储不同推理阶段时间消耗的字典
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        # 任务类型，标识为 "segment"
        self.task = "segment"
    @property
    def keys(self):
        """Returns a list of keys for accessing metrics."""
        # 返回用于访问指标的键列表，用于对象检测和语义分割模型的评估
        return [
            "metrics/precision(B)",    # 精度（Bounding Box）
            "metrics/recall(B)",       # 召回率（Bounding Box）
            "metrics/mAP50(B)",        # 平均精度 (mAP) @ IoU 50% （Bounding Box）
            "metrics/mAP50-95(B)",     # 平均精度 (mAP) @ IoU 50%-95% （Bounding Box）
            "metrics/precision(M)",    # 精度（Mask）
            "metrics/recall(M)",       # 召回率（Mask）
            "metrics/mAP50(M)",        # 平均精度 (mAP) @ IoU 50% （Mask）
            "metrics/mAP50-95(M)",     # 平均精度 (mAP) @ IoU 50%-95% （Mask）
        ]

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        # 返回边界框和分割结果的平均指标
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        # 返回指定类索引的分类结果
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        # 返回对象检测和语义分割模型的 mAP 分数
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        # 获取分割和边界框模型的适应性分数
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        # 边界框和掩膜具有相同的 ap_class_index
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        # 返回对象检测模型的评估结果
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    # 返回一个包含特定度量曲线的列表，用于访问特定度量曲线。
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",         # 精确率-召回率(B)
            "F1-Confidence(B)",            # F1-置信度(B)
            "Precision-Confidence(B)",     # 精确率-置信度(B)
            "Recall-Confidence(B)",        # 召回率-置信度(B)
            "Precision-Recall(M)",         # 精确率-召回率(M)
            "F1-Confidence(M)",            # F1-置信度(M)
            "Precision-Confidence(M)",     # 精确率-置信度(M)
            "Recall-Confidence(M)",        # 召回率-置信度(M)
        ]

    @property
    # 返回一个包含计算的性能指标和统计数据的字典。
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results
class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize the PoseMetrics class with directory path, class names, and plotting options."""
        # 调用父类的初始化方法，初始化基础类SegmentMetrics的属性
        super().__init__(save_dir, plot, names)
        # 设置实例属性：保存输出图表的目录路径
        self.save_dir = save_dir
        # 设置实例属性：是否保存检测和分割图表的标志
        self.plot = plot
        # 设置实例属性：用于在渲染时传递图表路径和数据的回调函数
        self.on_plot = on_plot
        # 设置实例属性：类名列表
        self.names = names
        # 设置实例属性：用于计算框检测指标的Metric类实例
        self.box = Metric()
        # 设置实例属性：用于计算姿势分割指标的Metric类实例
        self.pose = Metric()
        # 设置实例属性：存储推断不同阶段所花费时间的字典
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        # 设置实例属性：任务类型为姿势估计
        self.task = "pose"
    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_p (list): List of True Positive keypoints.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """

        # Calculate pose metrics per class and update PoseEvaluator
        results_pose = ap_per_class(
            tp_p,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        # Set the number of classes for pose evaluation
        self.pose.nc = len(self.names)
        # Update pose metrics with calculated results
        self.pose.update(results_pose)

        # Calculate box metrics per class and update BoxEvaluator
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        # Set the number of classes for box evaluation
        self.box.nc = len(self.names)
        # Update box metrics with calculated results
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self):
        """Return the mean results of box and pose."""
        # Return mean results of both box and pose evaluations
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """Return the class-wise detection results for a specific class i."""
        # Return class-wise detection results for class i from both box and pose evaluations
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose detections."""
        # Return mean average precision (mAP) per class for both box and pose detections
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and `pred` inputs."""
        # Compute classification metrics and speed using the `targets` and `pred` inputs for both box and pose
        return self.pose.fitness() + self.box.fitness()

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # Return a list of curves for accessing specific metrics curves
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        # Return dictionary of computed performance metrics and statistics for both box and pose
        return self.box.curves_results + self.pose.curves_results
class ClassifyMetrics(SimpleClass):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.
        fitness (float): The fitness of the model, which is equal to top-5 accuracy.
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        keys (List[str]): A list of keys for the results_dict.

    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        # 初始化 top1 和 top5 精度为 0
        self.top1 = 0
        self.top5 = 0
        # 初始化速度字典，包含各个步骤的时间，初始值都为 0.0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        # 设定任务类型为分类
        self.task = "classify"

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        # 合并预测结果和目标类别，以便计算准确率
        pred, targets = torch.cat(pred), torch.cat(targets)
        # 计算每个样本的正确性
        correct = (targets[:, None] == pred).float()
        # 计算 top-1 和 top-5 精度
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        """Returns mean of top-1 and top-5 accuracies as fitness score."""
        # 计算并返回 top-1 和 top-5 精度的平均值作为 fitness 分数
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        # 返回包含模型性能指标和 fitness 分数的字典
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        # 返回结果字典中的键列表
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # 返回一个空列表，用于访问特定的度量曲线
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # 返回一个空列表，用于访问特定的度量曲线
        return []


class OBBMetrics(SimpleClass):
    """Metrics for evaluating oriented bounding box (OBB) detection, see https://arxiv.org/pdf/2106.06072.pdf."""

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize an OBBMetrics instance with directory, plotting, callback, and class names."""
        # 初始化 OBBMetrics 实例，包括保存目录、绘图标志、回调函数和类名列表
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        # 初始化 Metric 类型的 box 属性
        self.box = Metric()
        # 初始化速度字典，包含各个步骤的时间，初始值都为 0.0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
    # 处理目标检测的预测结果并更新指标
    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        # 调用 ap_per_class 函数计算每个类别的平均精度等指标，返回结果列表，去掉前两个元素
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,  # 是否绘制结果的标志
            save_dir=self.save_dir,  # 结果保存目录
            names=self.names,  # 类别名称列表
            on_plot=self.on_plot,  # 是否在绘图时处理结果的标志
        )[2:]
        # 更新 self.box 对象的类别数
        self.box.nc = len(self.names)
        # 调用 self.box 对象的 update 方法，更新检测结果
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        # 返回用于访问特定指标的键列表
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        # 调用 self.box 对象的 mean_results 方法，计算检测到的物体的平均指标，返回包含这些指标的列表
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        # 调用 self.box 对象的 class_result 方法，返回指定类别 i 的性能评估结果
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        # 返回每个类别的平均精度 (mAP) 分数列表，由 self.box 对象的 maps 属性提供
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        # 返回 self.box 对象的 fitness 方法计算的适应度值
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        # 返回每个类别的平均精度索引，由 self.box 对象的 ap_class_index 属性提供
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        # 返回计算的性能指标和统计信息的字典，包括指标键列表和适应度值
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # 返回一个曲线列表，用于访问特定的指标曲线，这里返回一个空列表
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        # 返回一个曲线列表，用于访问特定的指标曲线，这里返回一个空列表
        return []
```