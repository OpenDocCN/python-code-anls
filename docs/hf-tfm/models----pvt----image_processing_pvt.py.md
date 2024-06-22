# `.\transformers\models\pvt\image_processing_pvt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 © 2023 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发是按“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
"""Pvt 的图像处理器类。"""

from typing import Dict, List, Optional, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理相关工具
from ...image_transforms import resize, to_channel_dimension_format  # 导入图像变换相关函数
from ...image_utils import (  # 导入图像处理相关工具函数
    IMAGENET_DEFAULT_MEAN,  # 导入 ImageNet 默认均值
    IMAGENET_DEFAULT_STD,  # 导入 ImageNet 默认标准差
    ChannelDimension,  # 导入通道维度枚举
    ImageInput,  # 导入图像输入类型
    PILImageResampling,  # 导入 PIL 图像重采样枚举
    infer_channel_dimension_format,  # 推断通道维度格式函数
    is_scaled_image,  # 判断是否为缩放图像函数
    make_list_of_images,  # 创建图像列表函数
    to_numpy_array,  # 将图像转换为 NumPy 数组函数
    valid_images,  # 判断图像是否有效函数
)
from ...utils import TensorType, logging  # 导入工具函数、模块

logger = logging.get_logger(__name__)  # 获取日志记录器


class PvtImageProcessor(BaseImageProcessor):  # 定义 PvtImageProcessor 类，继承自 BaseImageProcessor 类
    r"""
    构造一个 PVT 图像处理器。
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的（高度，宽度）尺寸调整为指定的 `(size["height"], size["width"])`。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            调整大小后输出图像的尺寸。可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            在调整图像大小时要使用的重采样滤波器。可以被 `preprocess` 方法中的 `resample` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定比例 `rescale_factor` 调整图像的尺度。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新调整图像尺寸，则要使用的比例因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            如果标准化图像，则要使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            如果标准化图像，则要使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        **kwargs,
    # 确定输入参数和返回值类型为 None
    ) -> None:
        # 调用父类的初始化方法，传递其他关键字参数
        super().__init__(**kwargs)
        # 如果 size 不为 None，则使用传入的 size，否则使用默认大小 {"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 获取 size 的大小字典
        size = get_size_dict(size)
        # 是否进行调整大小的标志
        self.do_resize = do_resize
        # 是否进行重新缩放的标志
        self.do_rescale = do_rescale
        # 是否进行标准化的标志
        self.do_normalize = do_normalize
        # 存储大小字典
        self.size = size
        # 重新采样方法
        self.resample = resample
        # 重新缩放因子
        self.rescale_factor = rescale_factor
        # 图像均值，如果为 None 则使用默认值 IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 图像标准差，如果为 None 则使用默认值 IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制而来
    def resize(
        # 输入图像的数组
        image: np.ndarray,
        # 目标大小的字典
        size: Dict[str, int],
        # 重新采样方法，默认为双线性插值
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        # 数据格式，默认为 None
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        # 将 `size` 参数转换为标准格式的大小字典
        size = get_size_dict(size)
        # 检查 `size` 字典是否包含必要的键，如果不包含则抛出 ValueError 异常
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 提取目标尺寸
        output_size = (size["height"], size["width"])
        # 调用 resize 函数对图像进行调整大小，并返回结果
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```