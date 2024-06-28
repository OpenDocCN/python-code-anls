# `.\models\nougat\image_processing_nougat.py`

```
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
"""Image processor class for Nougat."""

from typing import Dict, List, Optional, Union  # 导入类型提示所需的模块

import numpy as np  # 导入 NumPy 模块，用于处理数组和矩阵数据

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理所需的辅助函数和类
from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
    to_pil_image,
)  # 导入图像变换函数，包括调整大小、填充、转换格式等
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)  # 导入图像处理和验证函数
from ...utils import TensorType, logging  # 导入工具函数和日志记录模块
from ...utils.import_utils import is_cv2_available, is_vision_available  # 导入检查模块是否可用的函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


if is_cv2_available():
    pass  # 如果 OpenCV 可用，则无需额外导入任何模块

if is_vision_available():
    import PIL  # 如果 PIL 库可用，则导入 PIL 模块
    """
    Args:
        do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 896, "width": 672}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the images to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Image standard deviation.
    """

    # 定义模型输入名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_crop_margin: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        """
        Initialize the ImagePreprocessor object with optional parameters for image preprocessing.

        Args:
            do_crop_margin (bool, optional, default=True):
                Whether to crop the image margins.
            do_resize (bool, optional, default=True):
                Whether to resize the image to the specified `size`.
            size (Dict[str, int], optional, default={"height": 896, "width": 672}):
                Size of the image after resizing.
            resample (PILImageResampling, optional, default=PILImageResampling.BILINEAR):
                Resampling filter to use if resizing the image.
            do_thumbnail (bool, optional, default=True):
                Whether to resize the image using thumbnail method.
            do_align_long_axis (bool, optional, default=False):
                Whether to align the long axis of the image with the long axis of `size`.
            do_pad (bool, optional, default=True):
                Whether to pad the images to the largest image size in the batch.
            do_rescale (bool, optional, default=True):
                Whether to rescale the image by the specified scale `rescale_factor`.
            rescale_factor (int or float, optional, default=1/255):
                Scale factor to use if rescaling the image.
            do_normalize (bool, optional, default=True):
                Whether to normalize the image.
            image_mean (float or List[float], optional, default=IMAGENET_DEFAULT_MEAN):
                Mean to use if normalizing the image.
            image_std (float or List[float], optional, default=IMAGENET_DEFAULT_STD):
                Standard deviation of the image.
        """
    ) -> None:
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)

        # 如果 size 不为 None，则使用传入的 size，否则使用默认的高度和宽度
        size = size if size is not None else {"height": 896, "width": 672}
        # 将 size 转换为标准格式的字典
        size = get_size_dict(size)

        # 设置是否裁剪边距的标志
        self.do_crop_margin = do_crop_margin
        # 设置是否调整尺寸的标志
        self.do_resize = do_resize
        # 设置图像的尺寸
        self.size = size
        # 设置图像重采样的方式
        self.resample = resample
        # 设置是否生成缩略图的标志
        self.do_thumbnail = do_thumbnail
        # 设置是否沿长轴对齐的标志
        self.do_align_long_axis = do_align_long_axis
        # 设置是否填充图像的标志
        self.do_pad = do_pad
        # 设置是否重新缩放的标志
        self.do_rescale = do_rescale
        # 设置重新缩放的因子
        self.rescale_factor = rescale_factor
        # 设置是否归一化的标志
        self.do_normalize = do_normalize
        # 设置图像的均值，如果未提供则使用默认值 IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 设置图像的标准差，如果未提供则使用默认值 IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 设置有效的处理器关键字列表
        self._valid_processor_keys = [
            "images",
            "do_crop_margin",
            "do_resize",
            "size",
            "resample",
            "do_thumbnail",
            "do_align_long_axis",
            "do_pad",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    ) -> np.array:
        """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`np.array`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`):
                Value below which pixels are considered to be gray.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image. If unset, will use the inferred format from the
                input.
            input_data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the input image. If unset, will use the inferred format from the input.
        """
        # 如果未指定输入数据格式，则推断输入数据的通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 将输入图像转换为PIL图像对象，并根据输入数据格式转换
        image = to_pil_image(image, input_data_format=input_data_format)
        
        # 将PIL图像转换为灰度图像的NumPy数组，并将数据类型转换为uint8
        data = np.array(image.convert("L")).astype(np.uint8)
        
        # 计算灰度图像的最大值和最小值
        max_val = data.max()
        min_val = data.min()
        
        # 如果最大值等于最小值，说明图像中所有像素值相同，直接返回原始图像
        if max_val == min_val:
            image = np.array(image)
            # 根据输出数据格式和输入数据格式重新调整通道维度格式（如果输出数据格式不为空）
            image = (
                to_channel_dimension_format(image, data_format, input_data_format)
                if data_format is not None
                else image
            )
            return image
        
        # 对灰度图像进行归一化处理到0-255之间
        data = (data - min_val) / (max_val - min_val) * 255
        
        # 根据灰度阈值将图像分割成背景和前景
        gray = data < gray_threshold
        
        # 查找非零像素点的坐标
        coords = self.python_find_non_zero(gray)
        
        # 根据非零像素点的坐标计算边界框的位置和大小
        x_min, y_min, width, height = self.python_bounding_rect(coords)
        
        # 根据计算出的边界框裁剪原始图像
        image = image.crop((x_min, y_min, x_min + width, y_min + height))
        
        # 将裁剪后的图像转换为NumPy数组，并将数据类型转换为uint8
        image = np.array(image).astype(np.uint8)
        
        # 根据输出数据格式和输入数据格式重新调整通道维度格式
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
        
        # 根据输出数据格式和输入数据格式重新调整通道维度格式（如果输出数据格式不为空）
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )

        # 返回裁剪和调整通道维度格式后的图像
        return image

    # Copied from transformers.models.donut.image_processing_donut.DonutImageProcessor.align_long_axis
    def align_long_axis(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def align_long_axis(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The aligned image.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出图像的高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 如果输出宽度小于输出高度且输入宽度大于输入高度，或者输出宽度大于输出高度且输入宽度小于输入高度，则旋转图像
        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        # 如果指定了输出图像的数据格式，则将图像转换为指定格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回对齐后的图像
        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输出图像的高度和宽度
        output_height, output_width = size["height"], size["width"]
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # 计算宽度和高度的差值
        delta_width = output_width - input_width
        delta_height = output_height - input_height

        # 计算上、左、下、右四个方向的填充量
        pad_top = delta_height // 2
        pad_left = delta_width // 2
        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # 构造填充元组
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        # 返回填充后的图像
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)
    # 定义一个方法 `thumbnail`，用于生成缩略图
    def thumbnail(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出缩略图的高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 始终将图像大小调整为输入图像和输出大小的最小值
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        # 如果输入图像的大小已经符合要求，直接返回原图像
        if height == input_height and width == input_width:
            return image

        # 根据输入图像的高宽比例调整输出缩略图的高度和宽度
        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        # 调用resize方法进行图像缩放
        return resize(
            image,
            size=(height, width),
            resample=resample,
            reducing_gap=2.0,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 从transformers.models.donut.image_processing_donut.DonutImageProcessor.resize方法复制而来
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
    ) -> np.ndarray:
        """
        Resizes `image` to `(height, width)` specified by `size` using the PIL library.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 将 `size` 转换为标准的大小字典格式
        size = get_size_dict(size)
        # 获取 `size` 中较小的边长作为最短边
        shortest_edge = min(size["height"], size["width"])
        # 根据最短边和其他参数获取调整后的输出图像尺寸
        output_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
        )
        # 使用 resize 函数调整图像大小
        resized_image = resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return resized_image

    def preprocess(
        self,
        images: ImageInput,
        do_crop_margin: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```