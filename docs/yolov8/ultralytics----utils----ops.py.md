# `.\yolov8\ultralytics\utils\ops.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib  # 导入上下文管理器相关的模块
import math  # 导入数学函数模块
import re  # 导入正则表达式模块
import time  # 导入时间模块

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数模块

from ultralytics.utils import LOGGER  # 从ultralytics.utils中导入LOGGER对象
from ultralytics.utils.metrics import batch_probiou  # 从ultralytics.utils.metrics中导入batch_probiou函数


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```py
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t  # 初始化累计时间
        self.device = device  # 初始化设备
        self.cuda = bool(device and str(device).startswith("cuda"))  # 检查是否使用CUDA加速

    def __enter__(self):
        """Start timing."""
        self.start = self.time()  # 记录开始时间
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # 计算耗时
        self.t += self.dt  # 累加耗时到总时间

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"  # 返回累计的耗时信息

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)  # 同步CUDA流
        return time.time()  # 返回当前时间戳


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # 提取segment的xy坐标
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)  # 内部约束条件
    x = x[inside]  # 过滤符合约束条件的x坐标
    y = y[inside]  # 过滤符合约束条件的y坐标
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # 返回segment的最小和最大xy坐标，如果没有符合条件的点则返回全零数组


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): Shape of the original image (height, width).
        boxes (torch.Tensor): Bounding boxes in format xyxy.
        img0_shape (tuple): Shape of the new image (height, width).
        ratio_pad (tuple): Aspect ratio and padding.
        padding (bool): Whether to pad bounding boxes or not.
        xywh (bool): Whether the boxes are in xywh format or not. Defaults to False.
    """
    """
        Args:
            img1_shape (tuple): 目标图像的形状，格式为 (高度, 宽度)
            boxes (torch.Tensor): 图像中物体的边界框，格式为 (x1, y1, x2, y2)
            img0_shape (tuple): 原始图像的形状，格式为 (高度, 宽度)
            ratio_pad (tuple): 一个元组 (ratio, pad)，用于缩放边界框。如果未提供，则根据两个图像的大小差异计算 ratio 和 pad
            padding (bool): 如果为 True，则假设边界框基于 YOLO 样式增强的图像。如果为 False，则进行常规的重新缩放
            xywh (bool): 边界框格式是否为 xywh， 默认为 False
    
        Returns:
            boxes (torch.Tensor): 缩放后的边界框，格式为 (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # 如果未提供 ratio_pad，则从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比例 gain = 目标图像尺寸 / 原始图像尺寸
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),  # 计算宽度方向的填充量
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),  # 计算高度方向的填充量
        )
    else:
        gain = ratio_pad[0][0]  # 使用提供的 ratio_pad 中的缩放比例
        pad = ratio_pad[1]  # 使用提供的 ratio_pad 中的填充量
    
    if padding:
        boxes[..., 0] -= pad[0]  # 减去 x 方向的填充量
        boxes[..., 1] -= pad[1]  # 减去 y 方向的填充量
        if not xywh:
            boxes[..., 2] -= pad[0]  # 对于非 xywh 格式的边界框，再次减去 x 方向的填充量
            boxes[..., 3] -= pad[1]  # 对于非 xywh 格式的边界框，再次减去 y 方向的填充量
    
    boxes[..., :4] /= gain  # 缩放边界框坐标
    return clip_boxes(boxes, img0_shape)  # 调用 clip_boxes 函数，确保边界框在图像内部
# 执行非极大值抑制（NMS）操作，用于一组边界框，支持掩码和每个框多个标签。
def non_max_suppression(
    prediction,
    conf_thres=0.25,  # 置信度阈值，低于此阈值的框将被忽略
    iou_thres=0.45,  # IoU（交并比）阈值，用于判断重叠框之间是否合并
    classes=None,  # 类别列表，用于过滤特定类别的框
    agnostic=False,  # 是否忽略预测框的类别信息
    multi_label=False,  # 是否支持多标签输出
    labels=(),  # 标签列表，指定要保留的标签
    max_det=300,  # 最大检测框数
    nc=0,  # 类别数量（可选）
    max_time_img=0.05,  # 最大图像处理时间
    max_nms=30000,  # 最大NMS操作数
    max_wh=7680,  # 最大宽度和高度
    in_place=True,  # 是否就地修改
    rotated=False,  # 是否为旋转框
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    """
    # 如果预测为空，返回一个空的numpy数组
    if len(prediction) == 0:
        return np.empty((0,), dtype=np.int8)
    
    # 根据置信度对预测框进行降序排序
    sorted_idx = torch.argsort(prediction[:, 4], descending=True)
    prediction = prediction[sorted_idx]
    
    # 计算所有框两两之间的probiou得分矩阵，并取其上三角部分
    ious = batch_probiou(prediction, prediction).triu_(diagonal=1)
    
    # 根据IoU阈值进行非极大值抑制，保留符合条件的框索引
    pick = torch.nonzero(ious.max(dim=0)[0] < iou_thres).squeeze(-1)
    
    # 返回按照降序排列的被选框的索引
    return sorted_idx[pick]
    import torchvision  # 引入torchvision模块，用于加快“import ultralytics”的速度

    # 检查置信度阈值的有效性，必须在0到1之间
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    # 检查IoU阈值的有效性，必须在0到1之间
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # 如果prediction是一个列表或元组（例如YOLOv8模型在验证模式下的输出），选择推断输出部分
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # 选择推断输出

    # 如果指定了classes，则将其转换为与prediction设备相同的torch张量
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    # 如果prediction的最后一个维度为6，说明是端到端模型的输出（BNC格式，即1,300,6）
    if prediction.shape[-1] == 6:
        # 对每个预测结果进行置信度阈值过滤
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        # 如果指定了classes，则进一步根据classes进行过滤
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    # 获取batch size（BCN格式，即1,84,6300）
    bs = prediction.shape[0]
    # 如果未指定nc（类别数量），则根据prediction的形状推断类别数量
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    # 计算预测结果中的掩码数量
    nm = prediction.shape[1] - nc - 4  # number of masks
    
    # 确定掩码起始索引
    mi = 4 + nc  # mask start index
    
    # 根据置信度阈值确定候选项
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # 设置时间限制
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    
    # 若多标签设置为真，则每个框可能有多个标签（增加0.5ms/图像）
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # 调整预测结果的维度顺序，将最后两个维度互换
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    
    # 如果不是旋转框，根据需求将预测的边界框格式从xywh转换为xyxy
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy in-place modification
        else:
            # 在非原地操作时，将边界框和其他预测结果连接起来，转换为xyxy格式
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    # 记录当前时间
    t = time.time()
    
    # 初始化输出列表，每个元素都是一个空的张量，形状为(0, 6 + nm)，在指定设备上
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # 对每个预测结果进行遍历，xi是索引，x是预测结果
        # Apply constraints
        # 应用约束条件
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 对预测结果中的宽度和高度进行约束，将不满足条件的置为0

        x = x[xc[xi]]  # confidence
        # 根据置信度索引获取预测结果的子集

        # Cat apriori labels if autolabelling
        # 如果自动标注，合并先验标签
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)
            # 将先验标签与预测结果合并，形成新的预测结果

        # If none remain process next image
        # 如果没有剩余的预测结果，则处理下一张图像
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        # 检测矩阵，大小为nx6（xyxy坐标，置信度，类别）
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            # 如果支持多标签，根据置信度阈值筛选类别，并形成新的预测结果
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            # 否则，选择最高置信度的类别作为预测结果

        # Filter by class
        # 根据类别进行过滤
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
            # 如果指定了类别，只保留匹配指定类别的预测结果

        # Check shape
        # 检查预测结果的形状
        n = x.shape[0]  # number of boxes
        # n为盒子（边界框）的数量
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            # 如果盒子数量超过设定的最大NMS数量，则按置信度排序并保留前max_nms个盒子

        # Batched NMS
        # 批处理的非极大值抑制（NMS）
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
            # 如果启用了旋转NMS，对旋转边界框进行NMS处理
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            # 否则，对标准边界框进行NMS处理
        i = i[:max_det]  # limit detections
        # 限制最终的检测结果数量

        output[xi] = x[i]
        # 将处理后的预测结果存入输出中的对应位置
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
            # 如果超过了NMS处理时间限制，记录警告并跳出循环

    return output
def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # 获取当前 masks 的形状
    im1_shape = masks.shape
    
    # 如果当前 masks 形状与原始图片形状相同，则直接返回 masks，无需调整大小
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    
    # 如果未指定 ratio_pad，则根据 im0_shape 计算 gain 和 pad
    if ratio_pad is None:
        # 计算 gain，即缩放比例
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        # 计算 padding 的宽度和高度
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        pad = ratio_pad[1]  # 使用指定的 ratio_pad 中的 padding 值
    
    # 将 pad 转换为整数，表示上、左、下、右的边界
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
    
    # 如果 masks 的维度小于 2，则抛出异常
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    
    # 对 masks 进行裁剪，按照计算得到的边界进行裁剪
    masks = masks[top:bottom, left:right]
    
    # 将裁剪后的 masks 调整大小至原始图片大小
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    # 检查 masks 的维度是否为 2
    if len(masks.shape) == 2:
        # 如果是，添加一个额外的维度，使其变为三维
        masks = masks[:, :, None]

    # 返回处理后的 masks 变量
    return masks
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to normalized (x, y, width, height) format,
    relative to image dimensions and optionally clip the values.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): Width of the image. Defaults to 640.
        h (int): Height of the image. Defaults to 640.
        clip (bool): Whether to clip the normalized coordinates to [0, 1]. Defaults to False.
        eps (float): Epsilon value for numerical stability. Defaults to 0.0.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in normalized (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    half_w = w / 2.0
    half_h = h / 2.0
    y[..., 0] = (x[..., 0] + x[..., 2]) / (2 * w)  # center x normalized
    y[..., 1] = (x[..., 1] + x[..., 3]) / (2 * h)  # center y normalized
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width normalized
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height normalized

    if clip:
        y = torch.clamp(y, min=eps, max=1.0 - eps) if isinstance(y, torch.Tensor) else np.clip(y, eps, 1.0 - eps)

    return y
    # 将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height, normalized) 格式。其中 x, y, width 和 height 均已归一化至图像尺寸。
    
    Args:
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2)。
        w (int): 图像的宽度。默认为 640。
        h (int): 图像的高度。默认为 640。
        clip (bool): 如果为 True，则将边界框裁剪到图像边界内。默认为 False。
        eps (float): 边界框宽度和高度的最小值。默认为 0.0。
    
    Returns:
        y (np.ndarray | torch.Tensor): 格式为 (x, y, width, height, normalized) 的边界框坐标。
    """
    if clip:
        # 调用 clip_boxes 函数，将边界框 x 裁剪到图像边界内，边界为 (h - eps, w - eps)
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    # 根据输入 x 的类型创建与之相同类型的空数组 y，相比 clone/copy 操作更快
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    # 计算 x 中每个边界框的中心点 x 坐标，并将其归一化到图像宽度 w
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    # 计算 x 中每个边界框的中心点 y 坐标，并将其归一化到图像高度 h
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    # 计算 x 中每个边界框的宽度，并将其归一化到图像宽度 w
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    # 计算 x 中每个边界框的高度，并将其归一化到图像高度 h
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    # 返回格式为 (x, y, width, height, normalized) 的边界框坐标 y
    return y
def xywh2ltwh(x):
    """
    将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h]，其中 x1, y1 是左上角坐标。

    Args:
        x (np.ndarray | torch.Tensor): 输入张量，包含 xywh 格式的边界框坐标

    Returns:
        y (np.ndarray | torch.Tensor): 输出张量，包含 xyltwh 格式的边界框坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上角 x 坐标
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上角 y 坐标
    return y


def xyxy2ltwh(x):
    """
    将多个 [x1, y1, x2, y2] 格式的边界框转换为 [x1, y1, w, h] 格式，其中 xy1 是左上角，xy2 是右下角。

    Args:
        x (np.ndarray | torch.Tensor): 输入张量，包含 xyxy 格式的边界框坐标

    Returns:
        y (np.ndarray | torch.Tensor): 输出张量，包含 xyltwh 格式的边界框坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def ltwh2xywh(x):
    """
    将 [x1, y1, w, h] 格式的边界框转换为 [x, y, w, h] 格式，其中 xy1 是左上角，xy 是中心坐标。

    Args:
        x (torch.Tensor): 输入张量

    Returns:
        y (np.ndarray | torch.Tensor): 输出张量，包含 xywh 格式的边界框坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # 中心 x 坐标
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # 中心 y 坐标
    return y


def xyxyxyxy2xywhr(x):
    """
    将批量的方向边界框 (OBB) 从 [xy1, xy2, xy3, xy4] 格式转换为 [cx, cy, w, h, rotation] 格式。
    旋转角度的范围是从 0 到 90 度。

    Args:
        x (numpy.ndarray | torch.Tensor): 输入的角点数组 [xy1, xy2, xy3, xy4]，形状为 (n, 8)。

    Returns:
        (numpy.ndarray | torch.Tensor): 转换后的数据，形状为 (n, 5)，包含 [cx, cy, w, h, rotation] 格式。
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # 注意: 使用 cv2.minAreaRect 来获取准确的 xywhr 格式，
        # 特别是当数据加载器中的一些对象因增强而被裁剪时。
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    将批量的方向边界框 (OBB) 从 [cx, cy, w, h, rotation] 格式转换为 [xy1, xy2, xy3, xy4] 格式。
    旋转角度的范围应为 0 到 90 度。

    Args:
        x (numpy.ndarray | torch.Tensor): 输入的角点数组，形状为 (n, 5) 或 (b, n, 5)。

    Returns:
        (numpy.ndarray | torch.Tensor): 转换后的角点数组，形状为 (n, 4, 2) 或 (b, n, 4, 2)。
    """
    # 这个函数没有实现主体部分，因此不需要添加注释。
    pass
    # 根据输入的张量类型选择对应的数学函数库
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    # 提取张量 x 的中心坐标
    ctr = x[..., :2]
    # 提取张量 x 的宽度、高度和角度信息
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    # 计算角度的余弦和正弦值
    cos_value, sin_value = cos(angle), sin(angle)
    # 计算第一个向量 vec1
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    # 计算第二个向量 vec2
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    # 合并向量 vec1 的两个分量
    vec1 = cat(vec1, -1)
    # 合并向量 vec2 的两个分量
    vec2 = cat(vec2, -1)
    # 计算矩形的四个顶点
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    # 将四个顶点按行堆叠形成新的张量，并沿着倒数第二个维度堆叠
    return stack([pt1, pt2, pt3, pt4], -2)
def ltwh2xyxy(x):
    """
    将边界框从[x1, y1, w, h]转换为[x1, y1, x2, y2]，其中xy1为左上角，xy2为右下角。

    Args:
        x (np.ndarray | torch.Tensor): 输入的图像或张量

    Returns:
        y (np.ndarray | torch.Tensor): 边界框的xyxy坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # 计算宽度
    y[..., 3] = x[..., 3] + x[..., 1]  # 计算高度
    return y


def segments2boxes(segments):
    """
    将分段标签转换为框标签，即(cls, xy1, xy2, ...)转换为(cls, xywh)

    Args:
        segments (list): 分段列表，每个分段是一个点列表，每个点是一个包含x, y坐标的列表

    Returns:
        (np.ndarray): 边界框的xywh坐标
    """
    boxes = []
    for s in segments:
        x, y = s.T  # 提取分段的xy坐标
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # 计算xyxy坐标
    return xyxy2xywh(np.array(boxes))  # 转换为xywh坐标


def resample_segments(segments, n=1000):
    """
    将分段列表(samples,2)输入并将其上采样到每个n点的分段列表(samples,2)。

    Args:
        segments (list): 包含(samples,2)数组的列表，其中samples是分段中的点数。
        n (int): 要上采样到的点数，默认为1000。

    Returns:
        segments (list): 上采样后的分段列表。
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)  # 首尾相接，闭合分段
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # 插值获取上采样点
    return segments


def crop_mask(masks, boxes):
    """
    根据边界框裁剪掩模，并返回裁剪后的掩模。

    Args:
        masks (torch.Tensor): [n, h, w] 掩模张量
        boxes (torch.Tensor): [n, 4] 相对点形式的边界框坐标

    Returns:
        (torch.Tensor): 裁剪后的掩模
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # 分离边界框坐标
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # 行索引
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # 列索引

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    使用掩模头部的输出，将掩模应用于边界框。

    Args:
        protos: 未指定
        masks_in (torch.Tensor): [n, h, w] 掩模张量
        bboxes (torch.Tensor): [n, 4] 边界框坐标
        shape: 未指定
        upsample (bool): 是否上采样，默认为False

    Returns:
        unspecified
    """
    # 函数体未提供
    pass
    # 获取 protos 张量的形状信息，分别赋值给 c, mh, mw
    c, mh, mw = protos.shape  # CHW
    
    # 解构 shape 元组，获取输入图像的高度和宽度信息，分别赋值给 ih, iw
    ih, iw = shape
    
    # 计算每个 mask 的输出，通过 masks_in 与 protos 的矩阵乘法，再重新 reshape 成 [n, mh, mw] 的形状
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    
    # 计算宽度和高度的比率，用于将 bounding boxes 按比例缩放
    width_ratio = mw / iw
    height_ratio = mh / ih
    
    # 复制 bounding boxes 张量，按照比率调整左上角和右下角的坐标
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio
    
    # 裁剪 masks，根据 downsampled_bboxes 中的边界框信息进行裁剪，输出结果的形状保持为 CHW
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    
    # 如果 upsample 标志为 True，则对 masks 进行双线性插值，将其尺寸调整为 shape，最终形状为 [1, h, w]
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    
    # 返回 masks 张量中大于 0.0 的元素，即二值化后的二进制 mask 张量，形状为 [n, h, w]
    return masks.gt_(0.0)
# 定义函数 process_mask_native，处理原生掩模的逻辑
def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]，原型掩模的张量，形状为 [掩模维度, 高度, 宽度]
        masks_in (torch.Tensor): [n, mask_dim]，经 NMS 后的掩模张量，形状为 [n, 掩模维度]，n 为经过 NMS 后的掩模数量
        bboxes (torch.Tensor): [n, 4]，经 NMS 后的边界框张量，形状为 [n, 4]，n 为经过 NMS 后的掩模数量
        shape (tuple): 输入图像的尺寸 (高度, 宽度)

    Returns:
        masks (torch.Tensor): 处理后的掩模张量，形状为 [高度, 宽度, n]
    """
    c, mh, mw = protos.shape  # 获取原型掩模的通道数、高度、宽度
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # 计算掩模张量，进行上采样后裁剪到边界框大小
    masks = scale_masks(masks[None], shape)[0]  # 对掩模进行尺寸缩放
    masks = crop_mask(masks, bboxes)  # 根据边界框裁剪掩模
    return masks.gt_(0.0)  # 返回掩模张量，应用大于零的阈值处理


# 定义函数 scale_masks，将分段掩模尺寸缩放到指定形状
def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W)，掩模张量，形状为 (批量大小, 通道数, 高度, 宽度)
        shape (tuple): 目标高度和宽度
        padding (bool): 如果为 True，则假设边界框基于 YOLO 样式增强的图像。如果为 False，则进行常规尺寸缩放。

    Returns:
        masks (torch.Tensor): 缩放后的掩模张量
    """
    mh, mw = masks.shape[2:]  # 获取掩模张量的高度和宽度
    gain = min(mh / shape[0], mw / shape[1])  # 计算缩放比例 gain = 旧尺寸 / 新尺寸
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # 计算高度和宽度的填充值

    if padding:
        pad[0] /= 2  # 宽度填充减半
        pad[1] /= 2  # 高度填充减半

    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # 计算顶部和左侧填充位置
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))  # 计算底部和右侧填充位置
    masks = masks[..., top:bottom, left:right]  # 对掩模进行裁剪

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # 使用双线性插值对掩模进行尺寸缩放
    return masks


# 定义函数 scale_coords，将图像 1 的分割坐标缩放到图像 0 的尺寸
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): 坐标所在图像的尺寸。
        coords (torch.Tensor): 需要缩放的坐标，形状为 n,2。
        img0_shape (tuple): 应用分割的目标图像的尺寸。
        ratio_pad (tuple): 图像尺寸与填充图像尺寸的比例。
        normalize (bool): 如果为 True，则将坐标归一化到 [0, 1] 范围内。默认为 False。
        padding (bool): 如果为 True，则假设边界框基于 YOLO 样式增强的图像。如果为 False，则进行常规尺寸缩放。

    Returns:
        coords (torch.Tensor): 缩放后的坐标。
    """
    if ratio_pad is None:  # 如果没有指定比例，则根据图像 0 的尺寸计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比例 gain = 旧尺寸 / 新尺寸
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 计算高度和宽度的填充值
    else:
        gain = ratio_pad[0][0]  # 获取填充比例的缩放增益
        pad = ratio_pad[1]  # 获取填充值

    if padding:
        coords[..., 0] -= pad[0]  # 减去 x 方向的填充值
        coords[..., 1] -= pad[1]  # 减去 y 方向的填充值

    coords[..., 0] /= gain  # 根据缩放增益进行 x 坐标缩放
    coords[..., 1] /= gain  # 根据缩放增益进行 y 坐标缩放
    coords = clip_coords(coords, img0_shape)  # 调用 clip_coords 函数对坐标进行裁剪
    # 如果 normalize 参数为 True，则进行坐标归一化处理
    if normalize:
        # 将所有坐标点的 x 值除以图像宽度，实现 x 坐标的归一化
        coords[..., 0] /= img0_shape[1]  # width
        # 将所有坐标点的 y 值除以图像高度，实现 y 坐标的归一化
        coords[..., 1] /= img0_shape[0]  # height
    # 返回归一化后的坐标数组
    return coords
# Regularize rotated boxes in range [0, pi/2].
def regularize_rboxes(rboxes):
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge and angle if h >= w
    w_ = torch.where(w > h, w, h)  # Determine the maximum edge length
    h_ = torch.where(w > h, h, w)  # Determine the minimum edge length
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi  # Adjust angle if height is greater than width
    return torch.stack([x, y, w_, h_, t], dim=-1)  # Stack the regularized boxes


# It takes a list of masks(n,h,w) and returns a list of segments(n,xy)
def masks2segments(masks, strategy="largest"):
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        # Find contours in the mask image
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


# Convert a batch of FP32 torch tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.
def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


# Cleans a string by replacing special characters with underscore _
def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
```