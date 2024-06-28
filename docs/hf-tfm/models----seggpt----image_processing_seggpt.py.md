# `.\models\seggpt\image_processing_seggpt.py`

```
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for SegGPT."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_channel_dimension_axis,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, logging, requires_backends


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


# See https://arxiv.org/pdf/2212.02499.pdf  at 3.1 Redefining Output Spaces as "Images" - Semantic Segmentation from PAINTER paper
# Taken from https://github.com/Abdullah-Meda/Painter/blob/main/Painter/data/coco_semseg/gen_color_coco_panoptic_segm.py#L31
# 根据给定的类别数目构建调色板，返回一个颜色元组列表
def build_palette(num_labels: int) -> List[Tuple[int, int]]:
    base = int(num_labels ** (1 / 3)) + 1
    margin = 256 // base

    # 假设类别索引0代表背景，映射到黑色
    color_list = [(0, 0, 0)]
    # 生成调色板列表
    for location in range(num_labels):
        num_seq_r = location // base**2
        num_seq_g = (location % base**2) // base
        num_seq_b = location % base

        R = 255 - num_seq_r * margin
        G = 255 - num_seq_g * margin
        B = 255 - num_seq_b * margin

        color_list.append((R, G, B))

    return color_list


# 获取图像的通道数，根据输入数据格式和图像数组
def get_num_channels(image: np.ndarray, input_data_format: ChannelDimension) -> int:
    if image.ndim == 2:
        return 0

    channel_idx = get_channel_dimension_axis(image, input_data_format)
    return image.shape[channel_idx]


# 将掩码转换为RGB图像，使用指定的调色板（可选），输入数据格式和输出数据格式
def mask_to_rgb(
    mask: np.ndarray,
    palette: Optional[List[Tuple[int, int]]] = None,
    input_data_format: Optional[ChannelDimension] = None,
    data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    # 如果未指定输入数据格式并且掩码维度大于2，则推断输入数据格式
    if input_data_format is None and mask.ndim > 2:
        input_data_format = infer_channel_dimension_format(mask)

    # 如果未指定输出数据格式，则使用输入数据格式作为输出数据格式
    data_format = data_format if data_format is not None else input_data_format

    # 获取掩码的通道数
    num_channels = get_num_channels(mask, input_data_format)
    # 如果输入图像的通道数为 3，调用函数将掩码转换为指定通道格式，如果指定格式不为空；否则直接返回掩码
    if num_channels == 3:
        return to_channel_dimension_format(mask, data_format, input_data_format) if data_format is not None else mask

    # 如果调色板不为空，则处理彩色掩码
    if palette is not None:
        # 获取掩码的高度和宽度
        height, width = mask.shape

        # 创建一个全零的 RGB 掩码，形状为 (3, height, width)，数据类型为无符号 8 位整数
        rgb_mask = np.zeros((3, height, width), dtype=np.uint8)

        # 获取掩码中唯一的类别值
        classes_in_mask = np.unique(mask)

        # 遍历每个类别
        for class_idx in classes_in_mask:
            # 获取当前类别对应的 RGB 值
            rgb_value = palette[class_idx]

            # 创建当前类别的二值掩码，并扩展一个通道维度
            class_mask = (mask == class_idx).astype(np.uint8)
            class_mask = np.expand_dims(class_mask, axis=-1)

            # 将当前类别的 RGB 掩码计算出来，并移动通道维度到最前面
            class_rgb_mask = class_mask * np.array(rgb_value)
            class_rgb_mask = np.moveaxis(class_rgb_mask, -1, 0)

            # 将当前类别的 RGB 掩码加到总的 RGB 掩码上
            rgb_mask += class_rgb_mask.astype(np.uint8)

        # 将 RGB 掩码限制在 [0, 255] 范围内，并转换为无符号 8 位整数类型
        rgb_mask = np.clip(rgb_mask, 0, 255).astype(np.uint8)

    else:
        # 如果调色板为空，则将单通道掩码复制为三通道，形成灰度到 RGB 的映射
        rgb_mask = np.repeat(mask[None, ...], 3, axis=0)

    # 返回处理后的 RGB 掩码，如果指定通道格式不为空，则转换为指定格式；否则直接返回 RGB 掩码
    return (
        to_channel_dimension_format(rgb_mask, data_format, input_data_format) if data_format is not None else rgb_mask
    )
    r"""
    Constructs a SegGpt image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    # List of model input names expected by the SegGpt model
    model_input_names = ["pixel_values"]

    # Initialize the SegGptImageProcessor class with various parameters
    def __init__(
        self,
        do_resize: bool = True,  # Whether to resize images by default
        size: Optional[Dict[str, int]] = None,  # Default size for image resizing
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # Default resampling method
        do_rescale: bool = True,  # Whether to rescale images by default
        rescale_factor: Union[int, float] = 1 / 255,  # Default rescaling factor
        do_normalize: bool = True,  # Whether to normalize images by default
        image_mean: Optional[Union[float, List[float]]] = None,  # Default image mean for normalization
        image_std: Optional[Union[float, List[float]]] = None,  # Default image standard deviation for normalization
        **kwargs,  # Additional keyword arguments
    ) -> None:
        super().__init__(**kwargs)
        # 设置大小，如果未指定则使用默认大小
        size = size if size is not None else {"height": 448, "width": 448}
        # 根据给定的大小获取标准化后的尺寸字典
        size = get_size_dict(size)
        # 是否执行调整大小操作
        self.do_resize = do_resize
        # 是否执行重新缩放操作
        self.do_rescale = do_rescale
        # 是否执行标准化操作
        self.do_normalize = do_normalize
        # 图像的大小
        self.size = size
        # 重采样方法
        self.resample = resample
        # 重新缩放因子
        self.rescale_factor = rescale_factor
        # 图像均值，如果未指定则使用默认值
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 图像标准差，如果未指定则使用默认值
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def get_palette(self, num_labels: int) -> List[Tuple[int, int]]:
        """Build a palette to map the prompt mask from a single channel to a 3 channel RGB.

        Args:
            num_labels (`int`):
                Number of classes in the segmentation task (excluding the background).

        Returns:
            `List[Tuple[int, int]]`: Palette to map the prompt mask from a single channel to a 3 channel RGB.
        """
        # 调用 build_palette 函数创建用于将单通道掩码映射到3通道RGB的调色板
        return build_palette(num_labels)

    def mask_to_rgb(
        self,
        image: np.ndarray,
        palette: Optional[List[Tuple[int, int]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Convert a mask to RGB format.

        Args:
            image (`np.ndarray`):
                Mask to convert to RGB format. If the mask is already in RGB format, it will be passed through.
            palette (`List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Palette to use to convert the mask to RGB format. If unset, the mask is duplicated across the channel
                dimension.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The mask in RGB format.
        """
        # 调用 mask_to_rgb 函数将掩码转换为RGB格式的图像
        return mask_to_rgb(
            image,
            palette=palette,
            data_format=data_format,
            input_data_format=input_data_format,
        )
    # 定义 resize 方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
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
        # 获取调整后的目标尺寸字典
        size = get_size_dict(size)
        # 检查 size 字典是否包含 "height" 和 "width" 键
        if "height" not in size or "width" not in size:
            # 如果缺少必要的键，则引发 ValueError 异常
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 设定输出图像的尺寸元组
        output_size = (size["height"], size["width"])
        # 调用 resize 函数来实际执行图像的调整大小操作
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    # 定义私有方法 _preprocess_step，用于预处理步骤
    def _preprocess_step(
        self,
        images: ImageInput,
        is_mask: bool = False,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        # 预处理步骤，可能包括图像大小调整、缩放、归一化等操作
        pass

    # 定义公共方法 preprocess，用于执行预处理操作
    def preprocess(
        self,
        images: Optional[ImageInput] = None,
        prompt_images: Optional[ImageInput] = None,
        prompt_masks: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        num_labels: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        # 执行预处理，可能包括图像操作及其它参数的设置
        pass

    # 定义语义分割后处理方法 post_process_semantic_segmentation
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None, num_labels: Optional[int] = None
    ):
        # 对语义分割模型输出进行后处理，可能涉及到结果尺寸调整和标签数处理
        pass
```