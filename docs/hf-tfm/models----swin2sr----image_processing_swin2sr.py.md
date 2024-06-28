# `.\models\swin2sr\image_processing_swin2sr.py`

```
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
"""Image processor class for Swin2SR."""

from typing import Optional, Union

import numpy as np

# 导入基础的图像处理工具和转换函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import get_image_size, pad, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入日志记录工具
from ...utils import TensorType, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义图像处理器类 Swin2SRImageProcessor，继承自 BaseImageProcessor
class Swin2SRImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Swin2SR image processor.

    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
    """

    # 模型输入名称列表
    model_input_names = ["pixel_values"]

    # 初始化方法
    def __init__(
        self,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        pad_size: int = 8,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化各参数
        self.do_rescale = do_rescale            # 是否进行图像缩放
        self.rescale_factor = rescale_factor    # 缩放因子，默认为 1/255
        self.do_pad = do_pad                    # 是否进行填充
        self.pad_size = pad_size                # 填充尺寸
        self._valid_processor_keys = [          # 可接受的处理器关键字列表
            "images",
            "do_rescale",
            "rescale_factor",
            "do_pad",
            "pad_size",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 图像填充方法
    def pad(
        self,
        image: np.ndarray,
        size: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,

        # 实现图像填充操作，接受以下参数：
        #   - image: 待填充的图像数组
        #   - size: 填充尺寸
        #   - data_format: 输出数据格式（通道维度格式），可选
        #   - input_data_format: 输入数据格式（通道维度格式），可选
    ):
        """
        Pad an image to make the height and width divisible by `size`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`int`):
                The size to make the height and width divisible by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The padded image.
        """
        # 获取输入图像的原始高度和宽度
        old_height, old_width = get_image_size(image, input_data_format)
        # 计算需要填充的高度和宽度
        pad_height = (old_height // size + 1) * size - old_height
        pad_width = (old_width // size + 1) * size - old_width

        # 调用 pad 函数进行填充操作
        return pad(
            image,
            ((0, pad_height), (0, pad_width)),  # 在高度和宽度两个维度上进行填充
            mode="symmetric",  # 使用对称模式进行填充
            data_format=data_format,  # 指定输出图像的通道维度格式
            input_data_format=input_data_format,  # 指定输入图像的通道维度格式
        )

    def preprocess(
        self,
        images: ImageInput,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```