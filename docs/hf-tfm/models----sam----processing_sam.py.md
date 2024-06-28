# `.\models\sam\processing_sam.py`

```
# coding=utf-8
# 设置文件编码为 UTF-8，确保支持多语言字符

# 版权声明和许可证信息
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor class for SAM.
"""
# 导入深拷贝函数和类型提示模块
from copy import deepcopy
from typing import Optional, Union

# 导入 NumPy 库
import numpy as np

# 导入处理工具混合类、批编码类和工具函数
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available

# 如果可用，导入 PyTorch 库
if is_torch_available():
    import torch

# 如果可用，导入 TensorFlow 库
if is_tf_available():
    import tensorflow as tf

# SAM 处理器类，继承自处理器混合类
class SamProcessor(ProcessorMixin):
    r"""
    Constructs a SAM processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`SamProcessor`] offers all the functionalities of [`SamImageProcessor`]. See the docstring of
    [`~SamImageProcessor.__call__`] for more information.

    Args:
        image_processor (`SamImageProcessor`):
            An instance of [`SamImageProcessor`]. The image processor is a required input.
    """

    # 类属性：包含的属性列表
    attributes = ["image_processor"]
    # 类属性：图像处理器类名
    image_processor_class = "SamImageProcessor"

    # 初始化方法，接受图像处理器实例作为参数
    def __init__(self, image_processor):
        # 调用父类的初始化方法
        super().__init__(image_processor)
        # 设置当前处理器为图像处理器
        self.current_processor = self.image_processor
        # 设置点填充值为 -10
        self.point_pad_value = -10
        # 设置目标尺寸为图像处理器定义的最长边尺寸
        self.target_size = self.image_processor.size["longest_edge"]

    # 对象调用方法，用于处理输入数据
    def __call__(
        self,
        images=None,
        segmentation_maps=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    def _pad_points_and_labels(self, input_points, input_labels):
        """
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
        # Determine the maximum number of points among all input batches
        expected_nb_points = max([point.shape[0] for point in input_points])
        processed_input_points = []
        # Iterate over each batch of points and labels
        for i, point in enumerate(input_points):
            # If the number of points in the current batch is less than the maximum,
            # pad with zeros up to the maximum number of points
            if point.shape[0] != expected_nb_points:
                point = np.concatenate(
                    [point, np.zeros((expected_nb_points - point.shape[0], 2)) + self.point_pad_value], axis=0
                )
                # Append padding values to the corresponding labels array
                input_labels[i] = np.append(input_labels[i], [self.point_pad_value])
            processed_input_points.append(point)
        input_points = processed_input_points
        return input_points, input_labels

    def _normalize_coordinates(
        self, target_size: int, coords: np.ndarray, original_size, is_bounding_box=False
    ):
        """
        Normalize coordinates based on the target size and original size.
        """
        # Depending on whether the coordinates are for bounding boxes or not, calculate the scaling factor
        if is_bounding_box:
            scale = max(original_size) / float(target_size)
        else:
            scale = float(target_size) / max(original_size)
        # Normalize coordinates by scaling
        coords[:, 0::2] *= scale
        coords[:, 1::2] *= scale
        return coords
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        # 从参数 original_size 中获取原始图像的高度和宽度
        old_h, old_w = original_size
        # 使用 self.image_processor._get_preprocess_shape 方法获取预处理后的图像尺寸
        new_h, new_w = self.image_processor._get_preprocess_shape(original_size, longest_edge=target_size)
        # 深拷贝坐标数组，并将其转换为浮点数类型
        coords = deepcopy(coords).astype(float)

        # 如果 is_bounding_box 为 True，则将坐标数组重新整形为 (N, 2, 2) 的形式
        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        # 将坐标数组中 x 轴坐标缩放至新的宽度比例
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        # 将坐标数组中 y 轴坐标缩放至新的高度比例
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        # 如果 is_bounding_box 为 True，则将坐标数组重新整形为 (N, 4) 的形式
        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        # 返回处理后的坐标数组
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
        # 如果 input_points 不为 None，则进行下列处理
        if input_points is not None:
            # 如果 input_points 具有 "numpy" 属性，说明是 Torch 或 TF 张量，将其转换为 numpy.ndarray 再转换为 list
            if hasattr(input_points, "numpy"):  # Checks for TF or Torch tensor
                input_points = input_points.numpy().tolist()

            # 检查 input_points 是否为有效格式，必须是浮点数列表的列表
            if not isinstance(input_points, list) or not isinstance(input_points[0], list):
                raise ValueError("Input points must be a list of list of floating points.")
            # 将每个 input_points 转换为 numpy 数组
            input_points = [np.array(input_point) for input_point in input_points]
        else:
            input_points = None

        # 如果 input_labels 不为 None，则进行下列处理
        if input_labels is not None:
            # 如果 input_labels 具有 "numpy" 属性，将其转换为 list
            if hasattr(input_labels, "numpy"):
                input_labels = input_labels.numpy().tolist()

            # 检查 input_labels 是否为有效格式，必须是整数列表的列表
            if not isinstance(input_labels, list) or not isinstance(input_labels[0], list):
                raise ValueError("Input labels must be a list of list integers.")
            # 将每个 input_labels 转换为 numpy 数组
            input_labels = [np.array(label) for label in input_labels]
        else:
            input_labels = None

        # 如果 input_boxes 不为 None，则进行下列处理
        if input_boxes is not None:
            # 如果 input_boxes 具有 "numpy" 属性，将其转换为 list
            if hasattr(input_boxes, "numpy"):
                input_boxes = input_boxes.numpy().tolist()

            # 检查 input_boxes 是否为有效格式，必须是浮点数列表的列表的列表
            if (
                not isinstance(input_boxes, list)
                or not isinstance(input_boxes[0], list)
                or not isinstance(input_boxes[0][0], list)
            ):
                raise ValueError("Input boxes must be a list of list of list of floating points.")
            # 将每个 input_boxes 转换为 numpy 数组，并指定数据类型为 np.float32
            input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
        else:
            input_boxes = None

        # 返回处理后的 input_points, input_labels, input_boxes
        return input_points, input_labels, input_boxes

    @property
    def model_input_names(self):
        # 获取 self.image_processor 中的 model_input_names 属性，并去重后返回为列表
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))

    def post_process_masks(self, *args, **kwargs):
        # 调用 self.image_processor 中的 post_process_masks 方法，将参数传递并返回其结果
        return self.image_processor.post_process_masks(*args, **kwargs)
```