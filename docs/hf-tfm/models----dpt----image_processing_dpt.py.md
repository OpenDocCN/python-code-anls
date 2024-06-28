# `.\models\dpt\image_processing_dpt.py`

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
"""Image processor class for DPT."""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_torch_available,
    is_torch_tensor,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


if is_torch_available():
    import torch

if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


def get_resize_output_image_size(
    input_image: np.ndarray,
    output_size: Union[int, Iterable[int]],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Calculate the output size for resizing an image while optionally constraining to multiples and maintaining aspect ratio.

    Args:
        input_image (np.ndarray): The input image as a NumPy array.
        output_size (Union[int, Iterable[int]]): Desired output size for resizing.
        keep_aspect_ratio (bool): Whether to maintain the aspect ratio of the input image.
        multiple (int): Constraint to resize dimensions to multiples of this value.
        input_data_format (Optional[Union[str, ChannelDimension]], optional):
            Format of the input image data. Defaults to None.

    Returns:
        Tuple[int, int]: Output height and width after resizing.
    """

    def constraint_to_multiple_of(val, multiple, min_val=0, max_val=None):
        """
        Helper function to constrain a value to be a multiple of another value within specified bounds.

        Args:
            val (float): Value to constrain.
            multiple (int): Constraint value.
            min_val (int, optional): Minimum value constraint. Defaults to 0.
            max_val (int, optional): Maximum value constraint. Defaults to None.

        Returns:
            float: Constrained value.
        """
        x = round(val / multiple) * multiple

        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

    # Convert output_size to a tuple if it's an integer
    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    # Get dimensions of the input image
    input_height, input_width = get_image_size(input_image, input_data_format)
    output_height, output_width = output_size

    # Calculate scaling factors for height and width
    scale_height = output_height / input_height
    scale_width = output_width / input_width

    # Adjust scaling factors if maintaining aspect ratio
    if keep_aspect_ratio:
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width  # Fit width
        else:
            scale_width = scale_height  # Fit height

    # Calculate new height and width constrained to multiples
    new_height = constraint_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constraint_to_multiple_of(scale_width * input_width, multiple=multiple)

    return (new_height, new_width)


class DPTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DPT image processor.
    
    This class extends BaseImageProcessor and provides methods specific to the DPT image processing.
    """
    # 定义了多个参数，用于图像预处理过程中的控制和配置
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）尺寸。可以被 `preprocess` 中的 `do_resize` 覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 384}`):
            调整后的图像尺寸。可以被 `preprocess` 中的 `size` 覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            定义调整图像时要使用的重采样滤波器。可以被 `preprocess` 中的 `resample` 覆盖。
        keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
            如果为 `True`，则调整图像尺寸以保持宽高比最大可能的大小。可以被 `preprocess` 中的 `keep_aspect_ratio` 覆盖。
        ensure_multiple_of (`int`, *optional*, defaults to 1):
            如果 `do_resize` 为 `True`，则调整图像尺寸为此值的倍数。可以被 `preprocess` 中的 `ensure_multiple_of` 覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的比例因子 `rescale_factor` 进行图像缩放。可以被 `preprocess` 中的 `do_rescale` 覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果进行图像缩放，使用的缩放因子。可以被 `preprocess` 中的 `rescale_factor` 覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以被 `preprocess` 中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果进行图像归一化，使用的均值。这是一个浮点数或与图像通道数相等长度的浮点数列表。可以被 `preprocess` 中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果进行图像归一化，使用的标准差。这是一个浮点数或与图像通道数相等长度的浮点数列表。可以被 `preprocess` 中的 `image_std` 参数覆盖。
        do_pad (`bool`, *optional*, defaults to `False`):
            是否应用中心填充。这在与 DPT 结合使用的 DINOv2 论文中被引入。
        size_divisor (`int`, *optional*):
            如果 `do_pad` 为 `True`，则将图像维度填充为此值的倍数。这在与 DPT 结合使用的 DINOv2 论文中被引入。
    """

    # 定义了模型输入的名称，这是一个包含单个元素 "pixel_values" 的列表
    model_input_names = ["pixel_values"]
    # 初始化方法，用于实例化对象时进行初始化设置
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图片大小调整的标志，默认为True
        size: Dict[str, int] = None,  # 图片大小的字典，包含"height"和"width"两个键，默认为384x384
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图片调整大小时的重采样方法，默认为双三次插值
        keep_aspect_ratio: bool = False,  # 是否保持图片宽高比的标志，默认为False
        ensure_multiple_of: int = 1,  # 调整后的图片尺寸需为此值的倍数，默认为1
        do_rescale: bool = True,  # 是否对图片进行重新缩放的标志，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图片缩放因子，将像素值缩放到[0, 1]区间，默认为1/255
        do_normalize: bool = True,  # 是否对图片进行归一化处理的标志，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图片归一化的均值，默认为ImageNet数据集的标准均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图片归一化的标准差，默认为ImageNet数据集的标准差
        do_pad: bool = False,  # 是否对图片进行填充的标志，默认为False
        size_divisor: int = None,  # 图片调整后尺寸需为此值的倍数，默认为None
        **kwargs,  # 其他可选参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果未提供size参数，则使用默认值384x384
        size = size if size is not None else {"height": 384, "width": 384}
        # 根据提供的size参数获取标准化后的尺寸字典
        size = get_size_dict(size)
        # 初始化对象的各个属性
        self.do_resize = do_resize  # 是否进行图片大小调整的标志
        self.size = size  # 图片大小的字典
        self.keep_aspect_ratio = keep_aspect_ratio  # 是否保持图片宽高比的标志
        self.ensure_multiple_of = ensure_multiple_of  # 调整后的图片尺寸需为此值的倍数
        self.resample = resample  # 图片调整大小时的重采样方法
        self.do_rescale = do_rescale  # 是否对图片进行重新缩放的标志
        self.rescale_factor = rescale_factor  # 图片缩放因子
        self.do_normalize = do_normalize  # 是否对图片进行归一化处理的标志
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 图片归一化的均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 图片归一化的标准差
        self.do_pad = do_pad  # 是否对图片进行填充的标志
        self.size_divisor = size_divisor  # 图片调整后尺寸需为此值的倍数
        # 验证处理器可接受的键列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "keep_aspect_ratio",
            "ensure_multiple_of",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "size_divisor",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    ) -> np.ndarray:
        """
        Resize an image to target size `(size["height"], size["width"])`. If `keep_aspect_ratio` is `True`, the image
        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is
        set, the image is resized to a size that is a multiple of this value.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Target size of the output image.
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, the image is resized while preserving its aspect ratio.
            ensure_multiple_of (`int`, *optional*, defaults to 1):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the output image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size)  # 调用函数 `get_size_dict` 将 `size` 参数转换为标准尺寸字典
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size.keys()}")

        output_size = get_resize_output_image_size(
            image,
            output_size=(size["height"], size["width"]),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
            input_data_format=input_data_format,
        )
        # 调用 `get_resize_output_image_size` 函数计算输出图像的尺寸，并确保尺寸为 `size` 的倍数
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def pad_image(
        self,
        image: np.array,
        size_divisor: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,  # 允许额外的关键字参数
    ):
        """
        Pad an image to ensure its dimensions are divisible by `size_divisor`.

        Args:
            image (`np.array`):
                Image to pad.
            size_divisor (`int`):
                The divisor that the dimensions of the padded image should be divisible by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the output image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            **kwargs:
                Additional keyword arguments to be passed to the `resize` function.

        Returns:
            `np.ndarray`: Padded image.
        """
    ):
        """
        Center pad an image to be a multiple of `multiple`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size_divisor (`int`):
                The width and height of the image will be padded to a multiple of this number.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        def _get_pad(size, size_divisor):
            """
            Calculate padding sizes for an image dimension to be a multiple of `size_divisor`.

            Args:
                size (`int`): Original size of the image dimension.
                size_divisor (`int`): The width or height will be padded to a multiple of this number.

            Returns:
                tuple: Left and right padding sizes.
            """
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        
        height, width = get_image_size(image, input_data_format)

        pad_size_left, pad_size_right = _get_pad(height, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)

        return pad(image, ((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)), data_format=data_format)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: int = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        size_divisor: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess images according to specified transformations.

        Args:
            images (ImageInput): Input images to be preprocessed.
            do_resize (bool, optional): Whether to resize the images.
            size (int, optional): Size to which images should be resized.
            keep_aspect_ratio (bool, optional): Whether to maintain aspect ratio during resizing.
            ensure_multiple_of (int, optional): Ensure image dimensions are multiples of this number.
            resample (PILImageResampling, optional): Resampling method for resizing.
            do_rescale (bool, optional): Whether to rescale image pixel values.
            rescale_factor (float, optional): Factor to rescale image pixel values.
            do_normalize (bool, optional): Whether to normalize image pixel values.
            image_mean (float or List[float], optional): Mean values for image normalization.
            image_std (float or List[float], optional): Standard deviation values for image normalization.
            do_pad (bool, optional): Whether to pad images.
            size_divisor (int, optional): Pad image dimensions to be multiples of this number.
            return_tensors (str or TensorType, optional): Desired tensor type for output images.
            data_format (ChannelDimension, optional): Output image channel format.
            input_data_format (str or ChannelDimension, optional): Input image channel format.

            **kwargs: Additional keyword arguments for preprocessing.

        Returns:
            Preprocessed images according to the specified transformations.
        """
        # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation with Beit->DPT
    # 后处理语义分割模型输出，将[`DPTForSemanticSegmentation`]的输出转换为语义分割地图。仅支持PyTorch。

    # 获取模型输出中的逻辑回归值
    logits = outputs.logits

    # 调整逻辑回归值的大小并计算语义分割地图
    if target_sizes is not None:
        # 检查目标大小列表长度是否与逻辑回归值的批次维度相匹配
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        # 如果目标大小是PyTorch张量，则转换为NumPy数组
        if is_torch_tensor(target_sizes):
            target_sizes = target_sizes.numpy()

        # 初始化语义分割结果列表
        semantic_segmentation = []

        # 对每个样本的逻辑回归值进行处理
        for idx in range(len(logits)):
            # 调整逻辑回归值的大小，使用双线性插值方法
            resized_logits = torch.nn.functional.interpolate(
                logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            # 计算每个像素点的语义类别（最大概率对应的类别）
            semantic_map = resized_logits[0].argmax(dim=0)
            # 将处理后的语义分割地图添加到结果列表中
            semantic_segmentation.append(semantic_map)
    else:
        # 若未指定目标大小，则直接计算每个样本的语义分割结果
        semantic_segmentation = logits.argmax(dim=1)
        semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

    # 返回所有样本的语义分割结果列表
    return semantic_segmentation
```