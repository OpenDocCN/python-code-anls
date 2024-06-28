# `.\models\glpn\image_processing_glpn.py`

```py
# 设置脚本编码为 UTF-8，确保支持中文和其他非 ASCII 字符
# 版权声明，指出代码版权归 The HuggingFace Inc. 团队所有，保留一切权利
#
# 根据 Apache 许可证 2.0 版本使用本文件，详细条款可参见 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按“原样”提供，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取具体的使用条款和限制条件
"""GLPN 的图像处理类。"""

from typing import List, Optional, Union

import numpy as np  # 导入 NumPy 库用于数组操作
import PIL.Image  # 导入 PIL 库用于图像处理

from ...image_processing_utils import BaseImageProcessor, BatchFeature  # 导入基础图像处理工具和批量特征处理
from ...image_transforms import resize, to_channel_dimension_format  # 导入图像调整大小和转换通道维度的函数
from ...image_utils import (  # 导入图像工具函数，包括通道维度，PIL 图像重采样等
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging  # 导入 Tensor 类型和日志记录工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class GLPNImageProcessor(BaseImageProcessor):
    r"""
    构建 GLPN 图像处理器。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的 (height, width) 尺寸，将它们舍入到最接近的 `size_divisor` 的倍数。
            可以在 `preprocess` 中通过 `do_resize` 覆盖。
        size_divisor (`int`, *optional*, defaults to 32):
            当 `do_resize` 为 `True` 时，图像被调整大小，使其高度和宽度舍入到最接近的 `size_divisor` 的倍数。
            可以在 `preprocess` 中通过 `size_divisor` 覆盖。
        resample (`PIL.Image` resampling filter, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，要使用的重采样滤波器。可以在 `preprocess` 中通过 `resample` 覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否应用缩放因子（使像素值在 0 到 1 之间）。可以在 `preprocess` 中通过 `do_rescale` 覆盖。
    """

    model_input_names = ["pixel_values"]  # 模型输入的名称为 "pixel_values"

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        resample=PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        **kwargs,
    # 初始化图像处理器对象，设置是否调整大小、是否重新缩放、大小除数、重采样方法
    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = False,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        **kwargs,
    ) -> None:
        # 设置是否执行大小调整
        self.do_resize = do_resize
        # 设置是否执行重新缩放
        self.do_rescale = do_rescale
        # 设置大小除数
        self.size_divisor = size_divisor
        # 设置重采样方法
        self.resample = resample
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置有效的图像处理器关键字列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size_divisor",
            "resample",
            "do_rescale",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 调整图像大小，将图像的高度和宽度向下舍入到最接近的size_divisor的倍数
    def resize(
        self,
        image: np.ndarray,
        size_divisor: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        调整图像大小，将 (height, width) 维度向下舍入到最接近的 size_divisor 的倍数。

        如果图像的维度是 (3, 260, 170)，size_divisor 是 32，图像将被调整为 (3, 256, 160)。

        Args:
            image (`np.ndarray`):
                需要调整大小的图像。
            size_divisor (`int`):
                图像将被调整为其高度和宽度向下舍入到最接近的 `size_divisor` 的倍数。
            resample:
                调整图像时使用的 PIL.Image 重采样滤波器，例如 `PILImageResampling.BILINEAR`。
            data_format (`ChannelDimension` 或 `str`, *可选*):
                输出图像的通道维度格式。如果为 `None`，则使用输入图像的通道维度格式。可以是以下之一：
                - `ChannelDimension.FIRST`: 图像格式为 (num_channels, height, width)。
                - `ChannelDimension.LAST`: 图像格式为 (height, width, num_channels)。
            input_data_format (`ChannelDimension` 或 `str`, *可选*):
                输入图像的通道维度格式。如果未设置，则从输入图像推断通道维度格式。可以是以下之一：
                - `"channels_first"` 或 `ChannelDimension.FIRST`: 图像格式为 (num_channels, height, width)。
                - `"channels_last"` 或 `ChannelDimension.LAST`: 图像格式为 (height, width, num_channels)。

        Returns:
            `np.ndarray`: 调整大小后的图像。
        """
        # 获取图像的高度和宽度
        height, width = get_image_size(image, channel_dim=input_data_format)
        # 将高度和宽度向下舍入到最接近的 size_divisor 的倍数
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        # 调用 resize 函数进行图像调整
        image = resize(
            image,
            (new_h, new_w),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return image
    # 定义预处理方法，用于处理图像数据
    def preprocess(
        self,
        # 图像参数：可以是单张 PIL 图像、张量、或它们的列表
        images: Union["PIL.Image.Image", TensorType, List["PIL.Image.Image"], List[TensorType]],
        # 是否进行调整大小的标志，可以为布尔值或 None
        do_resize: Optional[bool] = None,
        # 调整大小的尺寸除数，可以为整数或 None
        size_divisor: Optional[int] = None,
        # 重采样方法，可以为 None 或指定重采样方法
        resample=None,
        # 是否进行重新缩放的标志，可以为布尔值或 None
        do_rescale: Optional[bool] = None,
        # 返回的数据格式，可以是张量、字符串或它们的组合
        return_tensors: Optional[Union[TensorType, str]] = None,
        # 数据格式的通道维度，通常为首通道（FIRST）或其他
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式，可以是字符串或通道维度对象的组合
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数，用于灵活设置
        **kwargs,
```