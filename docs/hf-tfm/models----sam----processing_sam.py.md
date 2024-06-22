# `.\transformers\models\sam\processing_sam.py`

```py
# 设置编码为 UTF-8
# 版权声明
# 遵循 Apache 许可证 2.0 版本
# 你不得使用这个文件，除非符合许可证的规定
# 你可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不提供任何形式的担保或条件，明示或暗示。
# 请参阅许可证了解特定语言下的权限和限制
"""
SAM 的处理器类。
"""
# 从深拷贝中导入 deepcopy 函数
from copy import deepcopy
# 从 typing 中导入 Optional 和 Union
from typing import Optional, Union
# 导入 numpy 库，命名为 np
import numpy as np
# 从 processing_utils 中导入 ProcessorMixin
from ...processing_utils import ProcessorMixin
# 从 tokenization_utils_base 中导入 BatchEncoding
from ...tokenization_utils_base import BatchEncoding
# 从 utils 中导入 TensorType, is_tf_available, is_torch_available
from ...utils import TensorType, is_tf_available, is_torch_available

# 如果 TensorFlow 可用
if is_tf_available():
    # 导入 tensorflow 库，命名为 tf
    import tensorflow as tf

# 如果 PyTorch 可用
if is_torch_available():
    # 导入 torch 库
    import torch

# 定义 SAM 处理器类，继承自 ProcessorMixin
class SamProcessor(ProcessorMixin):
    r"""
    构造一个 SAM 处理器，将 SAM 图像处理器和 2D 点和边界框处理器包装在一个单独的处理器中。

    [`SamProcessor`] 提供了 [`SamImageProcessor`] 的所有功能。有关更多信息，请参阅 [`~SamImageProcessor.__call__`] 的文档字符串。

    Args:
        image_processor (`SamImageProcessor`):
            一个 [`SamImageProcessor`] 的实例。图像处理器是必需的输入。
    """

    # 定义属性列表，包含 image_processor
    attributes = ["image_processor"]
    # 设置 image_processor_class 为 "SamImageProcessor"
    image_processor_class = "SamImageProcessor"

    # 初始化方法
    def __init__(self, image_processor):
        # 调用父类的初始化方法
        super().__init__(image_processor)
        # 将当前处理器设置为图像处理器
        self.current_processor = self.image_processor
        # 设置点的填充值为 -10
        self.point_pad_value = -10
        # 设置目标大小为图像处理器的最长边大小
        self.target_size = self.image_processor.size["longest_edge"]

    # 调用方法
    def __call__(
        self,
        images=None,
        segmentation_maps=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        # 调用 SamImageProcessor 类的 __call__ 方法来为模型准备图像。如果提供了2D点和边界框，则也为模型准备它们。
        encoding_image_processor = self.image_processor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors=return_tensors,
            **kwargs,
        )

        # 弹出在 forward 方法中没有使用但仍然使用的参数
        original_sizes = encoding_image_processor["original_sizes"]

        # 检查是否是 Torch 或 TF 张量
        if hasattr(original_sizes, "numpy"):
            original_sizes = original_sizes.numpy()

        # 检查和预处理2D点
        input_points, input_labels, input_boxes = self._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        # 标准化和转换输入
        encoding_image_processor = self._normalize_and_convert(
            encoding_image_processor,
            original_sizes,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors=return_tensors,
        )

        # 返回编码后的图像处理器
        return encoding_image_processor

    def _normalize_and_convert(
        self,
        encoding_image_processor,
        original_sizes,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors="pt",
    def _pad_points_and_labels(self, input_points, input_labels):
        r"""
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
        # 方法将2D点和标签填充到批次中最大点数
        expected_nb_points = max([point.shape[0] for point in input_points])
        processed_input_points = []
        for i, point in enumerate(input_points):
            if point.shape[0] != expected_nb_points:
                point = np.concatenate(
                    [point, np.zeros((expected_nb_points - point.shape[0], 2)) + self.point_pad_value], axis=0
                )
                input_labels[i] = np.append(input_labels[i], [self.point_pad_value])
            processed_input_points.append(point)
        input_points = processed_input_points
        return input_points, input_labels

    def _normalize_coordinates(
        self, target_size: int, coords: np.ndarray, original_size, is_bounding_box=False
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        # 获取原始图像的高度和宽度
        old_h, old_w = original_size
        # 计算目标尺寸下的新高度和宽度
        new_h, new_w = self.image_processor._get_preprocess_shape(original_size, longest_edge=target_size)
        # 深拷贝坐标，并将其转换为浮点型
        coords = deepcopy(coords).astype(float)

        # 如果是边界框，则对坐标进行重塑
        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        # 将坐标的 x 值按比例缩放到新宽度上
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        # 将坐标的 y 值按比例缩放到新高度上
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        # 如果是边界框，则将坐标再次重塑回原形式
        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        # 返回处理后的坐标
        return coords

    def _check_and_preprocess_points(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
    ):
        r"""
        Check and preprocesses the 2D points, labels and bounding boxes. It checks if the input is valid and if they
        are, it converts the coordinates of the points and bounding boxes. If a user passes directly a `torch.Tensor`,
        it is converted to a `numpy.ndarray` and then to a `list`.
        """
        # 检查并预处理输入的 2D 点、标签和边界框
        if input_points is not None:
            # 如果输入是张量，则转换为 numpy 数组再转换为列表
            if hasattr(input_points, "numpy"):  # Checks for TF or Torch tensor
                input_points = input_points.numpy().tolist()

            # 如果输入不是列表或第一个元素不是列表，则引发异常
            if not isinstance(input_points, list) or not isinstance(input_points[0], list):
                raise ValueError("Input points must be a list of list of floating points.")
            input_points = [np.array(input_point) for input_point in input_points]
        else:
            input_points = None

        # 检查并处理输入的标签
        if input_labels is not None:
            if hasattr(input_labels, "numpy"):
                input_labels = input_labels.numpy().tolist()

            if not isinstance(input_labels, list) or not isinstance(input_labels[0], list):
                raise ValueError("Input labels must be a list of list integers.")
            input_labels = [np.array(label) for label in input_labels]
        else:
            input_labels = None

        # 检查并处理输入的边界框
        if input_boxes is not None:
            if hasattr(input_boxes, "numpy"):
                input_boxes = input_boxes.numpy().tolist()

            if (
                not isinstance(input_boxes, list)
                or not isinstance(input_boxes[0], list)
                or not isinstance(input_boxes[0][0], list)
            ):
                raise ValueError("Input boxes must be a list of list of list of floating points.")
            input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
        else:
            input_boxes = None

        # 返回处理后的输入点、标签和边界框
        return input_points, input_labels, input_boxes

    @property
    def model_input_names(self):
        # 获取图像处理器的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 返回去重后的模型输入名称列表
        return list(dict.fromkeys(image_processor_input_names))

    def post_process_masks(self, *args, **kwargs):
        # 调用图像处理器的后处理掩码方法，并返回结果
        return self.image_processor.post_process_masks(*args, **kwargs)
```