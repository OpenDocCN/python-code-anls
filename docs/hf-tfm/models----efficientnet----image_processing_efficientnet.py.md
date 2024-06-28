# `.\models\efficientnet\image_processing_efficientnet.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Image processor class for EfficientNet.
"""

from typing import Dict, List, Optional, Union  # 导入需要的类型提示

import numpy as np  # 导入 NumPy 库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理相关的模块和函数
from ...image_transforms import rescale, resize, to_channel_dimension_format  # 导入图像变换相关函数
from ...image_utils import (  # 导入图像处理工具函数
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
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging  # 导入工具函数和模块

if is_vision_available():  # 如果视觉处理可用
    import PIL  # 导入 PIL 库用于图像处理

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class EfficientNetImageProcessor(BaseImageProcessor):
    r"""
    Constructs a EfficientNet image processor.
    
    This class inherits from BaseImageProcessor and is specialized for EfficientNet models.
    It provides methods for preprocessing images before feeding them into an EfficientNet model.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 346, "width": 346}`):
            Size of the image after `resize`. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling` filter, *optional*, defaults to 0):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in `preprocess`.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 289, "width": 289}`):
            Desired output size when applying center-cropping. Can be overridden by `crop_size` in `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        rescale_offset (`bool`, *optional*, defaults to `False`):
            Whether to rescale the image between [-scale_range, scale_range] instead of [0, scale_range]. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        include_top (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image again. Should be set to True if the inputs are used for image classification.
    """
    # 定义模型输入名称列表，只包含一个元素："pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化函数，设置图像处理器的各项参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行大小调整，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，键为"height"和"width"
        resample: PILImageResampling = PIL.Image.NEAREST,  # 图像调整大小时的重采样方法，默认为最近邻插值
        do_center_crop: bool = False,  # 是否进行中心裁剪，默认为False
        crop_size: Dict[str, int] = None,  # 裁剪大小的字典，键为"height"和"width"
        rescale_factor: Union[int, float] = 1 / 255,  # 图像缩放因子，默认为1/255
        rescale_offset: bool = False,  # 是否进行缩放偏移，默认为False
        do_rescale: bool = True,  # 是否进行缩放，默认为True
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可以是浮点数或浮点数列表
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，可以是浮点数或浮点数列表
        include_top: bool = True,  # 是否包含顶部处理，默认为True
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果未提供图像大小，则使用默认大小346x346
        size = size if size is not None else {"height": 346, "width": 346}
        # 根据给定的大小参数获取有效的尺寸字典
        size = get_size_dict(size)
        # 如果未提供裁剪大小，则使用默认大小289x289
        crop_size = crop_size if crop_size is not None else {"height": 289, "width": 289}
        # 根据给定的裁剪大小参数获取有效的裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 初始化对象的各个属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.rescale_offset = rescale_offset
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 如果未提供图像均值，则使用预设的ImageNet标准均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 如果未提供图像标准差，则使用预设的ImageNet标准标准差
        self.include_top = include_top
        # 初始化有效的处理器键列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "rescale_offset",
            "do_normalize",
            "image_mean",
            "image_std",
            "include_top",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor.resize复制，将PILImageResampling.BILINEAR更改为PILImageResampling.NEAREST
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据，numpy数组格式
        size: Dict[str, int],  # 调整后的图像大小字典，键为"height"和"width"
        resample: PILImageResampling = PILImageResampling.NEAREST,  # 图像调整大小时的重采样方法，默认为最近邻插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 图像数据格式，可以是字符串或通道维度
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的数据格式，可以是字符串或通道维度
        **kwargs,  # 其他关键字参数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: Optional[PILImageResampling] = PILImageResampling.NEAREST,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.NEAREST`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.NEAREST`.
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
        # 获取调整后的尺寸，确保 `size` 字典包含 `height` 和 `width` 键
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            # 如果 `size` 字典不包含 `height` 或 `width` 键，则抛出 ValueError 异常
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 计算输出的图像尺寸
        output_size = (size["height"], size["width"])
        # 调用 resize 函数对图像进行调整大小，并返回调整后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    ):
        """
        Rescale an image by a scale factor.

        If `offset` is `True`, the image has its values rescaled by `scale` and then offset by 1. If `scale` is
        1/127.5, the image is rescaled between [-1, 1].
            image = image * scale - 1

        If `offset` is `False`, and `scale` is 1/255, the image is rescaled between [0, 1].
            image = image * scale

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            offset (`bool`, *optional*):
                Whether to scale the image in both negative and positive directions.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 调用外部函数 `rescale` 对图像进行缩放
        rescaled_image = rescale(
            image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

        # 如果需要进行偏移处理
        if offset:
            # 将图像数值做偏移处理
            rescaled_image = rescaled_image - 1

        # 返回经过缩放和可能的偏移处理后的图像
        return rescaled_image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample=None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        rescale_offset: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        include_top: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```