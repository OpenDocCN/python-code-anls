# `.\models\blip\image_processing_blip.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for BLIP."""

from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


class BlipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BLIP image processor.
    """
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以在 `preprocess` 方法的 `do_resize` 参数中被覆盖。
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            调整后的输出图像尺寸。可以在 `preprocess` 方法的 `size` 参数中被覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            在调整图像大小时使用的重采样滤波器。仅在 `do_resize` 设置为 `True` 时有效。可以在 `preprocess` 方法的 `resample` 参数中被覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的缩放比例 `rescale_factor` 进行图像缩放。可以在 `preprocess` 方法的 `do_rescale` 参数中被覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果进行图像缩放，则使用的缩放因子。仅在 `do_rescale` 设置为 `True` 时有效。可以在 `preprocess` 方法的 `rescale_factor` 参数中被覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化处理。可以在 `preprocess` 方法的 `do_normalize` 参数中被覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果进行图像归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法的 `image_mean` 参数中被覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果进行图像归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法的 `image_std` 参数中被覆盖。
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            是否将图像转换为 RGB 格式。
    """

    # 定义模型输入的名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        """
        初始化方法，设置图像预处理参数。

        Args:
            do_resize (`bool`, *optional*, defaults to `True`): 是否将图像的（高度，宽度）尺寸调整为指定的 `size`。
                可以在 `preprocess` 方法的 `do_resize` 参数中被覆盖。
            size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`): 调整后的输出图像尺寸。
                可以在 `preprocess` 方法的 `size` 参数中被覆盖。
            resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
                在调整图像大小时使用的重采样滤波器。仅在 `do_resize` 设置为 `True` 时有效。
                可以在 `preprocess` 方法的 `resample` 参数中被覆盖。
            do_rescale (`bool`, *optional*, defaults to `True`): 是否按指定的缩放比例 `rescale_factor` 进行图像缩放。
                可以在 `preprocess` 方法的 `do_rescale` 参数中被覆盖。
            rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
                如果进行图像缩放，则使用的缩放因子。仅在 `do_rescale` 设置为 `True` 时有效。
                可以在 `preprocess` 方法的 `rescale_factor` 参数中被覆盖。
            do_normalize (`bool`, *optional*, defaults to `True`): 是否对图像进行归一化处理。
                可以在 `preprocess` 方法的 `do_normalize` 参数中被覆盖。
            image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
                如果进行图像归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。
                可以在 `preprocess` 方法的 `image_mean` 参数中被覆盖。
            image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
                如果进行图像归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。
                可以在 `preprocess` 方法的 `image_std` 参数中被覆盖。
            do_convert_rgb (`bool`, *optional*, defaults to `True`): 是否将图像转换为 RGB 格式。
        **kwargs: 其他未明确指定的参数，以字典形式传递。
        """
    # 初始化函数，继承父类的初始化方法，并设置一些参数的默认值
    ) -> None:
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)
        # 如果 size 参数不为 None，则使用它；否则设置默认的高度和宽度为 384
        size = size if size is not None else {"height": 384, "width": 384}
        # 根据给定的 size 字典，获取一个符合要求的尺寸字典，确保是正方形
        size = get_size_dict(size, default_to_square=True)

        # 设置对象的属性，用于控制是否进行图片缩放和缩放后的尺寸
        self.do_resize = do_resize
        self.size = size
        # 设置图像缩放时使用的重采样方法，默认为 BICUBIC
        self.resample = resample
        # 控制是否进行图像的线性缩放
        self.do_rescale = do_rescale
        # 图像缩放的因子
        self.rescale_factor = rescale_factor
        # 控制是否进行图像的标准化
        self.do_normalize = do_normalize
        # 图像标准化的均值，默认使用 OPENAI_CLIP_MEAN
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        # 图像标准化的标准差，默认使用 OPENAI_CLIP_STD
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        # 控制是否将图像转换为 RGB 格式
        self.do_convert_rgb = do_convert_rgb
        # 定义一个包含所有有效处理器键的列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制而来，用于调整图像大小，并将重采样方法从 BILINEAR 改为 BICUBIC
    def resize(
        self,
        # 图像的 ndarray 数组作为输入
        image: np.ndarray,
        # 目标尺寸的字典，包含高度和宽度
        size: Dict[str, int],
        # 图像的重采样方法，默认为 BICUBIC
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 数据格式参数，用于指定通道维度的表示方式
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据的格式参数，用于指定通道维度的表示方式
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
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
        # 调整输入的大小参数格式，确保其为字典形式
        size = get_size_dict(size)
        # 检查字典中是否包含必需的 height 和 width 键
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 根据输入的 size 字典获取目标输出的尺寸大小
        output_size = (size["height"], size["width"])
        # 调用 resize 函数对图像进行大小调整，返回调整后的图像数据
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
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```