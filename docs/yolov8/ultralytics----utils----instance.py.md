# `.\yolov8\ultralytics\utils\instance.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入必要的模块和库
from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np

# 从本地导入自定义的操作函数
from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh

# 定义一个辅助函数_ntuple，用于解析参数为可迭代对象或重复值
def _ntuple(n):
    """From PyTorch internals."""
    
    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    
    return parse

# 定义两个辅助函数，分别生成2元组和4元组
to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# 定义支持的边界框格式列表
# `xyxy` 表示左上角和右下角坐标
# `xywh` 表示中心点坐标和宽度、高度（YOLO格式）
# `ltwh` 表示左上角坐标和宽度、高度（COCO格式）
_formats = ["xyxy", "xywh", "ltwh"]

# 导出的类名列表
__all__ = ("Bboxes",)  # tuple or list

# 定义边界框类 Bboxes
class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        # 检查边界框格式是否有效
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        # 如果边界框是1维数组，则转换成2维数组
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        # 检查边界框数组维度为2
        assert bboxes.ndim == 2
        # 检查每个边界框的数组形状为(4,)
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        # 检查目标格式是否有效
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        # 如果当前格式与目标格式相同，则无需转换
        if self.format == format:
            return
        # 根据当前格式和目标格式选择相应的转换函数
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        # 执行转换，并更新边界框数组和格式
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """Return box areas."""
        # 计算每个边界框的面积
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    # 检查是否有任何边界框的值大于1.0
    assert (self.bboxes > 1.0).any()
    # 将所有边界框的 x 坐标和宽度进行归一化处理
    self.bboxes[:, 0::2] /= w
    # 将所有边界框的 y 坐标和高度进行归一化处理
    self.bboxes[:, 1::2] /= h
    # 设置标志，表示边界框已被归一化处理
    self.normalized = True

def mul(self, scale):
    """
    Args:
        scale (tuple | list | int): 四个坐标的缩放比例。
    """
    # 如果 scale 是一个单独的数值，则转换为包含四个相同值的元组
    if isinstance(scale, Number):
        scale = to_4tuple(scale)
    # 断言 scale 是元组或列表类型
    assert isinstance(scale, (tuple, list))
    # 断言 scale 的长度为四，即包含四个缩放比例
    assert len(scale) == 4
    # 将所有边界框的四个坐标分别乘以对应的缩放比例
    self.bboxes[:, 0] *= scale[0]
    self.bboxes[:, 1] *= scale[1]
    self.bboxes[:, 2] *= scale[2]
    self.bboxes[:, 3] *= scale[3]

def add(self, offset):
    """
    Args:
        offset (tuple | list | int): 四个坐标的偏移量。
    """
    # 如果 offset 是一个单独的数值，则转换为包含四个相同值的元组
    if isinstance(offset, Number):
        offset = to_4tuple(offset)
    # 断言 offset 是元组或列表类型
    assert isinstance(offset, (tuple, list))
    # 断言 offset 的长度为四，即包含四个偏移量
    assert len(offset) == 4
    # 将所有边界框的四个坐标分别加上对应的偏移量
    self.bboxes[:, 0] += offset[0]
    self.bboxes[:, 1] += offset[1]
    self.bboxes[:, 2] += offset[2]
    self.bboxes[:, 3] += offset[3]

def __len__(self):
    """返回边界框的数量。"""
    return len(self.bboxes)

@classmethod
def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
    """
    将一个 Bboxes 对象的列表或元组连接成一个单一的 Bboxes 对象。

    Args:
        boxes_list (List[Bboxes]): 要连接的 Bboxes 对象的列表。
        axis (int, optional): 沿着哪个轴连接边界框。默认为 0。

    Returns:
        Bboxes: 包含连接后的边界框的新 Bboxes 对象。

    Note:
        输入应为 Bboxes 对象的列表或元组。
    """
    # 断言 boxes_list 是列表或元组类型
    assert isinstance(boxes_list, (list, tuple))
    # 如果 boxes_list 为空，则返回一个空的 Bboxes 对象
    if not boxes_list:
        return cls(np.empty(0))
    # 断言 boxes_list 中的所有元素都是 Bboxes 对象
    assert all(isinstance(box, Bboxes) for box in boxes_list)

    # 如果 boxes_list 只包含一个元素，则直接返回这个元素
    if len(boxes_list) == 1:
        return boxes_list[0]
    # 使用 np.concatenate 将所有 Bboxes 对象中的边界框数组连接起来
    return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))
    # 定义一个特殊方法 __getitem__，用于通过索引获取特定的边界框或一组边界框。

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """

        # 如果索引是整数，返回一个包含单个边界框的新 Bboxes 对象
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].view(1, -1))

        # 对于其他类型的索引，直接获取对应的边界框数组
        b = self.bboxes[index]

        # 断言所得到的边界框数组是二维矩阵，否则抛出异常
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"

        # 返回一个新的 Bboxes 对象，其中包含选定的边界框数组
        return Bboxes(b)
class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        keypoints (ndarray, optional): An array of keypoints with shape [N, 17, 3]. Default is None.
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.

    Examples:
        ```py
        # Create an Instances object
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]])
        )
        ```py

    Note:
        The bounding box format is either 'xywh' or 'xyxy', and is determined by the `bbox_format` argument.
        This class does not perform input validation, and it assumes the inputs are well-formed.
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True) -> None:
        """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
        # Initialize internal bounding box handler with given format
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        # Set keypoints attribute
        self.keypoints = keypoints
        # Set normalized flag
        self.normalized = normalized
        # Set segments attribute
        self.segments = segments

    def convert_bbox(self, format):
        """Convert bounding box format."""
        # Delegate conversion to internal bounding box handler
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        # Retrieve areas of bounding boxes using internal handler
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """This might be similar with denormalize func but without normalized sign."""
        # Scale bounding boxes
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        # If only bbox scaling is requested, return early
        if bbox_only:
            return
        # Scale segments coordinates
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        # If keypoints exist, scale their coordinates as well
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h
    def denormalize(self, w, h):
        """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
        # 如果未进行归一化，则直接返回
        if not self.normalized:
            return
        # 缩放边界框（bounding boxes），分割（segments）和关键点（keypoints）到原始坐标
        self._bboxes.mul(scale=(w, h, w, h))
        # 对分割的 x 和 y 坐标进行反归一化
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        # 如果存在关键点数据，则对其 x 和 y 坐标进行反归一化
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        # 标记对象已经不再是归一化状态
        self.normalized = False

    def normalize(self, w, h):
        """Normalize bounding boxes, segments, and keypoints to image dimensions."""
        # 如果已经进行了归一化，则直接返回
        if self.normalized:
            return
        # 将边界框（bounding boxes），分割（segments）和关键点（keypoints）归一化到图像尺寸
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        # 标记对象已经处于归一化状态
        self.normalized = True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation."""
        # 断言对象未处于归一化状态，即只能使用绝对坐标添加填充
        assert not self.normalized, "you should add padding with absolute coordinates."
        # 添加填充到边界框（bounding boxes），分割（segments）和关键点（keypoints）
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> "Instances":
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        # 根据索引获取特定的实例或一组实例
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        # 返回一个新的 Instances 对象，包含所选的边界框（bounding boxes），分割（segments）和关键点（keypoints）
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        """Flips the coordinates of bounding boxes, segments, and keypoints vertically."""
        # 如果边界框的格式是 "xyxy"，则对应的顶部和底部坐标进行垂直翻转
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            # 否则直接对 y 坐标进行垂直翻转
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        # 对分割的 y 坐标进行垂直翻转
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            # 如果存在关键点数据，则对其 y 坐标进行垂直翻转
            self.keypoints[..., 1] = h - self.keypoints[..., 1]
    def fliplr(self, w):
        """Reverses the order of the bounding boxes and segments horizontally."""
        # 检查边界框格式是否为 "xyxy"
        if self._bboxes.format == "xyxy":
            # 复制边界框的 x1 和 x2 坐标
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            # 更新边界框的 x1 和 x2 坐标以反转水平方向
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            # 更新边界框的 x 坐标以反转水平方向
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        # 更新段的 x 坐标以反转水平方向
        self.segments[..., 0] = w - self.segments[..., 0]
        # 如果关键点不为 None，则更新关键点的 x 坐标以反转水平方向
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        """Clips bounding boxes, segments, and keypoints values to stay within image boundaries."""
        # 保存原始的边界框格式
        ori_format = self._bboxes.format
        # 转换边界框格式为 "xyxy"
        self.convert_bbox(format="xyxy")
        # 将边界框的 x 和 y 坐标限制在图像边界内
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        # 如果原始边界框格式不是 "xyxy"，则转换回原始格式
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        # 将段的 x 和 y 坐标限制在图像边界内
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        # 如果关键点不为 None，则将关键点的 x 和 y 坐标限制在图像边界内
        if self.keypoints is not None:
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self):
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height."""
        # 判断哪些边界框面积大于 0
        good = self.bbox_areas > 0
        # 如果存在面积为 0 的边界框，则移除它们
        if not all(good):
            self._bboxes = self._bboxes[good]
            # 如果段的长度不为 0，则移除与边界框对应的段
            if len(self.segments):
                self.segments = self.segments[good]
            # 如果关键点不为 None，则移除与边界框对应的关键点
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        # 返回保留的边界框索引列表
        return good

    def update(self, bboxes, segments=None, keypoints=None):
        """Updates instance variables."""
        # 更新边界框变量
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        # 如果提供了段变量，则更新段变量
        if segments is not None:
            self.segments = segments
        # 如果提供了关键点变量，则更新关键点变量
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """Return the length of the instance list."""
        # 返回边界框列表的长度
        return len(self.bboxes)
    # 定义一个类方法，用于将多个 Instances 对象连接成一个单一的 Instances 对象
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        Concatenates a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
        # 断言 instances_list 是一个列表或元组
        assert isinstance(instances_list, (list, tuple))
        # 如果 instances_list 为空列表，则返回一个空的 Instances 对象
        if not instances_list:
            return cls(np.empty(0))
        # 断言 instances_list 中的所有元素都是 Instances 对象
        assert all(isinstance(instance, Instances) for instance in instances_list)

        # 如果 instances_list 中只有一个元素，则直接返回该元素
        if len(instances_list) == 1:
            return instances_list[0]

        # 确定是否使用关键点
        use_keypoint = instances_list[0].keypoints is not None
        # 获取边界框格式
        bbox_format = instances_list[0]._bboxes.format
        # 获取是否使用了规范化的标志
        normalized = instances_list[0].normalized

        # 按指定轴连接边界框数组
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        # 按指定轴连接分割数组
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        # 如果使用关键点，则按指定轴连接关键点数组；否则设置为 None
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        # 返回一个新的 Instances 对象，包含连接后的边界框、分割和关键点（如果有）、边界框格式和规范化信息
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self):
        """Return bounding boxes."""
        # 返回私有成员变量 _bboxes 的 bboxes 属性
        return self._bboxes.bboxes
```