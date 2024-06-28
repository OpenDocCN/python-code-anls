# `.\models\mobilenet_v1\image_processing_mobilenet_v1.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 代码文件的版权声明，声明此代码版权归 HuggingFace Inc. 团队所有

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
# 引入 Apache License 2.0，指明本代码遵循的许可协议，允许自由使用、分发及修改

"""Image processor class for MobileNetV1."""
# MobileNetV1 图像处理类的定义

from typing import Dict, List, Optional, Union

import numpy as np  # 导入 NumPy 库

# 导入所需的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,  # 导入图像尺寸调整函数
    resize,  # 导入图像缩放函数
    to_channel_dimension_format,  # 导入通道维度格式转换函数
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入 ImageNet 标准均值
    IMAGENET_STANDARD_STD,  # 导入 ImageNet 标准标准差
    ChannelDimension,  # 导入通道维度类
    ImageInput,  # 导入图像输入类
    PILImageResampling,  # 导入 PIL 图像重采样枚举
    infer_channel_dimension_format,  # 推断通道维度格式的函数
    is_scaled_image,  # 判断是否为缩放图像的函数
    make_list_of_images,  # 将图像列表化的函数
    to_numpy_array,  # 将图像转换为 NumPy 数组的函数
    valid_images,  # 验证图像合法性的函数
    validate_kwargs,  # 验证关键字参数的函数
    validate_preprocess_arguments,  # 验证预处理参数的函数
)
from ...utils import TensorType, logging  # 导入 Tensor 类型及日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class MobileNetV1ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MobileNetV1 image processor.
    构造一个 MobileNetV1 图像处理器类。
    # 定义函数参数
    Args:
        # 是否调整图像的（高度，宽度）尺寸到指定的尺寸，默认为True
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        # 图像调整大小后的尺寸，默认为`{"shortest_edge": 256}`
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        # 图像调整大小时使用的重采样滤波器，默认为`PILImageResampling.BILINEAR`
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        # 是否在图像中心裁剪，默认为True
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in the
            `preprocess` method.
        # 应用中心裁剪时所需的输出大小，默认为`{"height": 224, "width": 224}`
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
            Can be overridden by the `crop_size` parameter in the `preprocess` method.
        # 是否按指定的比例对图像进行重新缩放，默认为True
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        # 如果重新缩放图像，使用的比例因子，默认为1/255
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        # 是否对图像进行正规化
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        # 如果对图像进行正规化，则使用的均值，默认为`IMAGENET_STANDARD_MEAN`
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        # 如果对图像进行正规化，则使用的标准差，默认为`IMAGENET_STANDARD_STD`
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    
    # 模型输入的名称为"pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化方法，设置图像处理器对象的各项属性
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志
        size: Optional[Dict[str, int]] = None,  # 图像调整后的大小
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整时的重采样方法
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 裁剪后的图像大小
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志
        rescale_factor: Union[int, float] = 1 / 255,  # 图像像素值缩放的因子
        do_normalize: bool = True,  # 是否进行图像标准化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像标准化的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准化的标准差
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果未提供调整大小的具体尺寸，使用默认值 {"shortest_edge": 256}
        size = size if size is not None else {"shortest_edge": 256}
        # 根据给定的尺寸字典获取尺寸信息，确保不会是方形
        size = get_size_dict(size, default_to_square=False)
        # 如果未提供裁剪大小的具体尺寸，使用默认值 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据给定的裁剪尺寸字典获取裁剪尺寸信息
        crop_size = get_size_dict(crop_size)
        # 初始化对象的属性
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
        # 设置有效的处理器关键字列表，用于后续的处理判断
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize方法复制而来
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据，类型为NumPy数组
        size: Dict[str, int],  # 调整后的图像尺寸字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像调整时的重采样方法
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式
        **kwargs,
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. Should contain either "shortest_edge" or "height" and "width".
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Determine if default square behavior should be applied
        default_to_square = True

        # Check if "shortest_edge" is specified in size dictionary
        if "shortest_edge" in size:
            # If "shortest_edge" is specified, resize the image based on it
            size = size["shortest_edge"]
            default_to_square = False
        # Check if both "height" and "width" are specified in size dictionary
        elif "height" in size and "width" in size:
            # If both dimensions are specified, resize the image based on them
            size = (size["height"], size["width"])
        else:
            # If neither "shortest_edge" nor ("height" and "width") are specified, raise an error
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        # Calculate the output size for resizing the image
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        # Perform the resizing operation on the image using the calculated output size
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
```