# `.\models\llava_next\image_processing_llava_next.py`

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
"""Image processor class for LLaVa-NeXT."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def divide_to_patches(image: np.array, patch_size: int, input_data_format) -> List[np.array]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    """
    patches = []
    # 获取输入图像的高度和宽度
    height, width = get_image_size(image, channel_dim=input_data_format)
    # 循环遍历图像并划分成指定大小的补丁
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                # 如果通道维度在最后，则按照 (行, 列) 的方式提取补丁
                patch = image[i : i + patch_size, j : j + patch_size]
            else:
                # 否则按照 (通道, 行, 列) 的方式提取补丁
                patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


def expand_to_square(image: np.array, background_color, input_data_format) -> np.array:
    """
    Expands an image to a square by adding a background color.

    Args:
        image (`np.array`):
            The input image to be expanded.
        background_color:
            The color value to be used for the background.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        np.array: The expanded square image.
    """
    # 获取输入图像的高度和宽度
    height, width = get_image_size(image, channel_dim=input_data_format)
    # 如果图像已经是正方形，则直接返回
    if width == height:
        return image
    elif width > height:
        # 如果宽度大于高度，需要在顶部和底部添加背景颜色，使其变为正方形
        result = np.ones((width, width, image.shape[2]), dtype=image.dtype) * background_color
        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        return result
    # 如果图像宽度小于高度，则在宽度方向上填充背景色，使其与高度相等的正方形图像
    else:
        # 创建一个与原图像相同高度和通道数的全 1 数组，并用背景色填充
        result = np.ones((height, height, image.shape[2]), dtype=image.dtype) * background_color
        # 将原图像嵌入到正方形图像中心区域，保持其原有的宽度
        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        # 返回处理后的正方形图像
        return result
# 计算调整后图像的尺寸，使其符合目标分辨率
def _get_patch_output_size(image, target_resolution, input_data_format):
    # 获取原始图像的高度和宽度
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    
    # 获取目标分辨率的高度和宽度
    target_height, target_width = target_resolution

    # 计算宽度和高度的缩放比例
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    # 根据缩放比例确定新的宽度和高度
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # 返回调整后的新高度和新宽度
    return new_height, new_width


class LlavaNextImageProcessor(BaseImageProcessor):
    r"""
    构造一个LLaVa-NeXT图像处理器。基于`CLIPImageProcessor`，结合了处理高分辨率图像的额外技术，
    此技术详见[LLaVa论文](https://arxiv.org/abs/2310.03744)。

    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        # 如果未提供size，则默认使用{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 将size参数转换为标准化的尺寸字典，不默认使用方形
        size = get_size_dict(size, default_to_square=False)
        
        # 如果未提供image_grid_pinpoints，则使用默认值[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        
        # 如果未提供crop_size，则默认使用{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 将crop_size参数转换为标准化的尺寸字典，默认使用方形
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 设置对象的各种属性
        self.do_resize = do_resize
        self.size = size
        self.image_grid_pinpoints = image_grid_pinpoints
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    # 从transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize复制而来，用于LLaVa
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
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

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
        default_to_square = True  # 默认按最短边缩放到指定尺寸，保持长宽比
        if "shortest_edge" in size:
            size = size["shortest_edge"]  # 将尺寸设置为最短边的长度
            default_to_square = False  # 不默认按正方形缩放
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])  # 将尺寸设置为给定的高度和宽度
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
            # 如果尺寸参数不包含最短边或者高度和宽度，则抛出数值错误异常

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 调用函数计算输出图像的大小

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        # 调用自身方法进行图像缩放操作，并返回缩放后的图像

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess images with optional resizing, cropping, rescaling, and normalization.

        Args:
            images (`ImageInput`):
                Input images to preprocess.
            do_resize (`bool`, *optional*):
                Whether to perform resizing.
            size (`Dict[str, int]`, *optional*):
                Target size for resizing.
            resample (`PILImageResampling`, *optional*):
                Resampling filter to use for resizing.
            do_center_crop (`bool`, *optional*):
                Whether to perform center cropping.
            crop_size (`int`, *optional*):
                Size for center cropping.
            do_rescale (`bool`, *optional*):
                Whether to perform rescaling.
            rescale_factor (`float`, *optional*):
                Factor for rescaling.
            do_normalize (`bool`, *optional*):
                Whether to perform normalization.
            image_mean (`float` or `List[float]`, *optional*):
                Mean value(s) for normalization.
            image_std (`float` or `List[float]`, *optional*):
                Standard deviation value(s) for normalization.
            data_format (`ChannelDimension`, *optional*):
                Channel dimension format of the images.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                Input channel dimension format.

        """
        # 该方法用于对图像进行预处理，包括可选的缩放、裁剪、重新缩放和归一化操作

    def _resize_for_patching(
        self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension
    ):
        """
        Resize an image for patching.

        Args:
            image (`np.array`):
                Image to resize.
            target_resolution (`tuple`):
                Target resolution (height, width) for resizing.
            resample:
                Resampling filter to use for resizing.
            input_data_format (`ChannelDimension`):
                Input channel dimension format.

        """
        # 该方法用于为图像裁剪调整大小操作
    ) -> np.array:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """
        # Calculate the new height and width for resizing
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image using specified parameters
        resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)

        return resized_image

    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        # Extract target height and width from target resolution
        target_height, target_width = target_resolution
        
        # Calculate new output size for resizing
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        # Calculate paste positions for padding
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Pad the image symmetrically
        padded_image = pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ):
        """
        Extract patches from an image based on specified grid points and patch size.
        """
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """
        # Check if grid_pinpoints is a list; raise error if not
        if not isinstance(grid_pinpoints, list):
            raise ValueError("grid_pinpoints must be a list of possible resolutions.")

        # Assign grid_pinpoints to possible_resolutions for clarity
        possible_resolutions = grid_pinpoints

        # Determine the size of the input image
        image_size = get_image_size(image, channel_dim=input_data_format)

        # Select the best resolution from possible_resolutions based on image_size
        best_resolution = select_best_resolution(image_size, possible_resolutions)

        # Resize the original image to best_resolution
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )

        # Pad the resized image for patching purposes
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        # Divide the padded image into patches of size patch_size
        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)

        # Ensure all patches are in the desired output data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]

        # Resize the original image to size specified by `size`
        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        # Combine the resized original image and processed patches into image_patches list
        image_patches = [resized_original_image] + patches

        # Return the list of image patches
        return image_patches
    # 图像预处理方法，用于对输入图像进行多种处理操作
    def preprocess(
        self,
        # 输入图像，可以是单张图像或图像列表
        images: ImageInput,
        # 是否进行调整图像大小的操作，默认为 None
        do_resize: bool = None,
        # 调整后的目标大小，字典格式，包含宽度和高度信息
        size: Dict[str, int] = None,
        # 图像网格定位点列表，用于特定的图像处理任务
        image_grid_pinpoints: List = None,
        # 重采样方法，例如最近邻法、双线性插值等
        resample: PILImageResampling = None,
        # 是否进行中心裁剪操作，默认为 None
        do_center_crop: bool = None,
        # 裁剪后的目标大小
        crop_size: int = None,
        # 是否进行图像重新缩放操作，默认为 None
        do_rescale: bool = None,
        # 重新缩放因子，控制缩放的比例
        rescale_factor: float = None,
        # 是否进行图像标准化操作，默认为 None
        do_normalize: bool = None,
        # 图像的均值，可以是单一值或者通道均值列表
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像的标准差，可以是单一值或者通道标准差列表
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否将图像转换为 RGB 格式，默认为 None
        do_convert_rgb: bool = None,
        # 返回的张量类型，例如字符串或张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式，控制通道的位置顺序
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        # 输入数据的格式描述，可以是字符串或通道维度对象
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
```