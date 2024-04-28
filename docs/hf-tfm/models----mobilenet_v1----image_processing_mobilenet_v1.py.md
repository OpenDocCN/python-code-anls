# `.\transformers\models\mobilenet_v1\image_processing_mobilenet_v1.py`

```
# 这是 MobileNetV1 图像处理器的代码
# 它包含了一些版权信息和许可证声明
# 它导入了一些必要的库和模块，包括 typing、numpy、image_processing_utils、image_transforms 和 image_utils 等
# 它定义了一个 MobileNetV1ImageProcessor 类，作为 BaseImageProcessor 类的子类

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
"""Image processor class for MobileNetV1."""

from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class MobileNetV1ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MobileNetV1 image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）尺寸到指定的 `size`。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            调整后图像的尺寸。图像的最短边被调整为 `size["shortest_edge"]`，最长边被调整以保持输入的长宽比。可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            如果调整图像大小，则使用的重采样滤波器。可以被 `preprocess` 方法中的 `resample` 参数覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪。如果输入尺寸小于任何一边的 `crop_size`，则用 0 填充图像，然后进行中心裁剪。可以被 `preprocess` 方法中的 `do_center_crop` 参数覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            应用中心裁剪时的期望输出尺寸。仅在 `do_center_crop` 设置为 `True` 时有效。可以被 `preprocess` 方法中的 `crop_size` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定比例 `rescale_factor` 对图像进行重新缩放。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，则使用的比例因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize:
            是否对图像进行标准化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果对图像进行标准化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果对图像进行标准化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """

    model_input_names = ["pixel_values"]
    # 初始化方法，用于初始化图像处理器对象
    def __init__(
        self,
        # 是否进行大小调整，默认为True
        do_resize: bool = True,
        # 图像大小，可选参数，默认为None
        size: Optional[Dict[str, int]] = None,
        # 重采样方法，默认为双线性插值
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        # 是否进行中心裁剪，默认为True
        do_center_crop: bool = True,
        # 裁剪尺寸，默认为None
        crop_size: Dict[str, int] = None,
        # 是否进行尺度变换，默认为True
        do_rescale: bool = True,
        # 尺度因子，默认为1/255
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否进行标准化，默认为True
        do_normalize: bool = True,
        # 图像均值，默认为None
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差，默认为None
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果size为None，则设置默认的图像大小字典
        size = size if size is not None else {"shortest_edge": 256}
        # 将size转换为标准的尺寸字典
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size为None，则设置默认的裁剪尺寸字典
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 将crop_size转换为标准的尺寸字典
        crop_size = get_size_dict(crop_size)
        # 初始化各参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    # 从CLIP源代码复制的resize方法
    def resize(
        self,
        # 待处理的图像
        image: np.ndarray,
        # 目标尺寸
        size: Dict[str, int],
        # 重采样方法，默认为双三次插值
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 数据格式，默认为None
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True  # 设置一个默认值，如果没有指定最短边的长度，则默认设置为正方形
        if "shortest_edge" in size:  # 检查是否在尺寸参数中指定了最短边的长度
            size = size["shortest_edge"]  # 如果指定了最短边的长度，则将尺寸参数设置为该长度
            default_to_square = False  # 取消默认设置为正方形的标志
        elif "height" in size and "width" in size:  # 如果没有指定最短边的长度，但指定了高度和宽度
            size = (size["height"], size["width"])  # 将尺寸参数设置为高度和宽度组成的元组
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")  # 如果尺寸参数中既没有指定最短边的长度，也没有指定高度和宽度，则引发值错误

        output_size = get_resize_output_image_size(  # 调用函数计算调整后的输出图像大小
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        return resize(  # 返回调整大小后的图像
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(  # 定义预处理函数
        self,
        images: ImageInput,  # 图像输入参数
        do_resize: Optional[bool] = None,  # 是否调整大小的标志
        size: Dict[str, int] = None,  # 输出图像的尺寸参数
        resample: PILImageResampling = None,  # 重采样滤波器类型
        do_center_crop: bool = None,  # 是否中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 裁剪尺寸参数
        do_rescale: Optional[bool] = None,  # 是否重新缩放的标志
        rescale_factor: Optional[float] = None,  # 重新缩放因子
        do_normalize: Optional[bool] = None,  # 是否归一化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量的格式
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 图像通道格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像通道格式
        **kwargs,  # 其他关键字参数
```