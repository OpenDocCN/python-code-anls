# `.\models\flava\image_processing_flava.py`

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
"""Image processor class for Flava."""

import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
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


# These values are taken from CLIP
FLAVA_IMAGE_MEAN = OPENAI_CLIP_MEAN
FLAVA_IMAGE_STD = OPENAI_CLIP_STD
FLAVA_CODEBOOK_MEAN = [0.0, 0.0, 0.0]
FLAVA_CODEBOOK_STD = [1.0, 1.0, 1.0]
LOGIT_LAPLACE_EPS: float = 0.1


# Inspired from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
        mask_group_max_aspect_ratio: float = None,
    ):
        # 如果输入大小不是元组，则将其转换为元组
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        # 初始化输入的高度和宽度
        self.height, self.width = input_size

        # 计算总的掩码片段数
        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches

        # 设定每个掩码组的最小和最大片段数
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches

        # 根据最小和最大纵横比计算对数纵横比的范围
        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))
    # 返回对象的字符串表示，描述了 MaskingGenerator 实例的参数和范围
    def __repr__(self):
        repr_str = "MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.mask_group_min_patches,
            self.mask_group_max_patches,
            self.total_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    # 返回生成器的高度和宽度
    def get_shape(self):
        return self.height, self.width

    # 执行掩码生成的核心方法，修改给定的掩码并返回修改的像素数
    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            # 随机确定目标区域的面积
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            # 随机生成长宽比
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            # 根据面积和长宽比计算高度和宽度
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            # 如果生成的掩码区域在合理范围内
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                # 计算当前掩码区域中已经被掩盖的像素数
                num_masked = mask[top : top + height, left : left + width].sum()
                # 如果新生成的掩盖区域与当前掩盖区域有重叠
                if 0 < height * width - num_masked <= max_mask_patches:
                    # 将新区域中未被掩盖的像素进行掩盖
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                # 如果有像素被掩盖，则结束循环
                if delta > 0:
                    break
        # 返回掩盖操作导致的像素数变化
        return delta

    # 生成器的调用方法，生成并返回掩码
    def __call__(self):
        # 创建一个与生成器相同形状的零矩阵作为掩码
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        # 循环生成掩码，直到达到指定的总掩盖像素数
        while mask_count < self.total_mask_patches:
            # 每次最多可以生成的掩码数
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)

            # 执行掩码生成，获取本次生成的掩码像素数
            delta = self._mask(mask, max_mask_patches)
            # 如果没有新的像素被掩盖，则结束生成过程
            if delta == 0:
                break
            else:
                mask_count += delta

        # 返回生成的掩码
        return mask
class FlavaImageProcessor(BaseImageProcessor):
    r"""
    构造一个 Flava 图像处理器。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        # 是否进行调整大小
        do_resize: bool = True,
        # 图像大小
        size: Dict[str, int] = None,
        # 重采样方法
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 是否进行中心裁剪
        do_center_crop: bool = True,
        # 裁剪大小
        crop_size: Dict[str, int] = None,
        # 是否进行重新缩放
        do_rescale: bool = True,
        # 缩放系数
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否进行归一化
        do_normalize: bool = True,
        # 图像均值
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, Iterable[float]]] = None,
        # Mask 相关参数
        return_image_mask: bool = False,
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: int = 16,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook 相关参数
        return_codebook_pixels: bool = False,
        codebook_do_resize: bool = True,
        codebook_size: bool = None,
        codebook_resample: int = PILImageResampling.LANCZOS,
        codebook_do_center_crop: bool = True,
        codebook_crop_size: int = None,
        codebook_do_rescale: bool = True,
        codebook_rescale_factor: Union[int, float] = 1 / 255,
        codebook_do_map_pixels: bool = True,
        codebook_do_normalize: bool = True,
        codebook_image_mean: Optional[Union[float, Iterable[float]]] = None,
        codebook_image_std: Optional[Union[float, Iterable[float]]] = None,
        **kwargs,
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        重写基类的 `from_dict` 方法，以确保在使用 from_dict 和 kwargs 创建图像处理器时更新参数，
        例如 `FlavaImageProcessor.from_pretrained(checkpoint, codebook_size=600)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "codebook_size" in kwargs:
            image_processor_dict["codebook_size"] = kwargs.pop("codebook_size")
        if "codebook_crop_size" in kwargs:
        image_processor_dict["codebook_crop_size"] = kwargs.pop("codebook_crop_size")
        return super().from_dict(image_processor_dict, **kwargs)

    @lru_cache()
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    # 返回一个 FlavaMaskingGenerator 实例，使用给定的参数初始化
    ) -> FlavaMaskingGenerator:
        return FlavaMaskingGenerator(
            input_size=input_size_patches,  # 设置输入大小为 input_size_patches
            total_mask_patches=total_mask_patches,  # 设置总掩蔽片段数为 total_mask_patches
            mask_group_min_patches=mask_group_min_patches,  # 设置掩蔽组最小片段数为 mask_group_min_patches
            mask_group_max_patches=mask_group_max_patches,  # 设置掩蔽组最大片段数为 mask_group_max_patches
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,  # 设置掩蔽组最小长宽比为 mask_group_min_aspect_ratio
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,  # 设置掩蔽组最大长宽比为 mask_group_max_aspect_ratio
        )

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 中复制的函数
    # 用于调整图像大小，使用 BICUBIC 插值算法，接受一个 np.ndarray 格式的图像数据
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据，格式为 np.ndarray
        size: Dict[str, int],  # 目标大小，以字典形式提供，包含 'height' 和 'width' 键
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 插值方法，默认为 BICUBIC
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据格式，可选参数
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选参数
        **kwargs,  # 其它可选参数
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
        size = get_size_dict(size)  # 获取处理后的尺寸字典
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # 设置输出图像的尺寸
        return resize(
            image,
            size=output_size,  # 调整图像大小至指定尺寸
            resample=resample,  # 使用指定的重采样滤波器，默认为双三次插值
            data_format=data_format,  # 设置输出图像的通道顺序格式
            input_data_format=input_data_format,  # 设置输入图像的通道顺序格式，默认从输入图像推断
            **kwargs,
        )

    def map_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Maps pixel values of an image using a specific constant.

        Args:
            image (`np.ndarray`):
                Input image.

        Returns:
            `np.ndarray`: Processed image with mapped pixel values.
        """
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_map_pixels: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        # 将图像转换为 numpy 数组
        image = to_numpy_array(image)

        if is_scaled_image(image) and do_rescale:
            # 如果图像已经缩放并且设定了重新缩放选项，则发出警告
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # 假设所有图像具有相同的通道维度格式
            input_data_format = infer_channel_dimension_format(image)

        if do_resize:
            # 如果需要调整大小，则调用 resize 方法
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            # 如果需要中心裁剪，则调用 center_crop 方法
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            # 如果需要重新缩放，则调用 rescale 方法
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            # 如果需要标准化，则调用 normalize 方法
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        if do_map_pixels:
            # 如果需要像素映射，则调用 map_pixels 方法
            image = self.map_pixels(image)

        if data_format is not None:
            # 如果指定了数据格式，则将图像转换为该格式的通道维度格式
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回预处理后的图像
        return image
    # 定义一个方法用于预处理输入的图像数据，包含多个参数用于控制不同的预处理步骤和参数设置
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        # Mask 相关参数
        return_image_mask: Optional[bool] = None,
        input_size_patches: Optional[int] = None,
        total_mask_patches: Optional[int] = None,
        mask_group_min_patches: Optional[int] = None,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: Optional[float] = None,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook 相关参数
        return_codebook_pixels: Optional[bool] = None,
        codebook_do_resize: Optional[bool] = None,
        codebook_size: Optional[Dict[str, int]] = None,
        codebook_resample: Optional[int] = None,
        codebook_do_center_crop: Optional[bool] = None,
        codebook_crop_size: Optional[Dict[str, int]] = None,
        codebook_do_rescale: Optional[bool] = None,
        codebook_rescale_factor: Optional[float] = None,
        codebook_do_map_pixels: Optional[bool] = None,
        codebook_do_normalize: Optional[bool] = None,
        codebook_image_mean: Optional[Iterable[float]] = None,
        codebook_image_std: Optional[Iterable[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
```