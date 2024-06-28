# `.\image_transforms.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .image_utils import (
    ChannelDimension,
    ImageInput,
    get_channel_dimension_axis,
    get_image_size,
    infer_channel_dimension_format,
)
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    requires_backends,
)

if is_vision_available():
    import PIL

    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax.numpy as jnp


def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))  # Reorder dimensions to put channels first
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))  # Reorder dimensions to put channels last
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Rescales the input `image` by a factor of `scale`.

    Args:
        image (`numpy.ndarray`):
            The image to be rescaled.
        scale (`float`):
            The scaling factor to be applied to the image.
        data_format (`ChannelDimension`, *optional*):
            The desired channel dimension format of the output image.
        dtype (`np.dtype`, *optional*):
            The desired data type of the output image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    # 按比例 `scale` 重新调整 `image` 的大小。

    # 检查输入参数 `image` 是否为 `np.ndarray` 类型，否则引发 ValueError 异常
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    # 将 `image` 按比例 `scale` 进行重新调整
    rescaled_image = image * scale

    # 如果提供了 `data_format` 参数，则将 `rescaled_image` 转换为指定的通道维度格式
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    # 将 `rescaled_image` 转换为指定的数据类型 `dtype`
    rescaled_image = rescaled_image.astype(dtype)

    # 返回重新调整大小后的图像 `rescaled_image`
    return rescaled_image
# 检查输入的图像是否需要在转换为 PIL 图像之前进行重新缩放
def _rescale_for_pil_conversion(image):
    if image.dtype == np.uint8:
        # 如果图像类型为 np.uint8，则无需重新缩放
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            # 如果图像的所有值都在 [0, 255] 范围内，则无需重新缩放
            do_rescale = False
        else:
            # 抛出异常，因为图像包含超出 [0, 255] 范围的值
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(0 <= image) and np.all(image <= 1):
        # 如果图像的所有值都在 [0, 1] 范围内，则需要重新缩放
        do_rescale = True
    else:
        # 抛出异常，因为图像包含超出 [0, 1] 范围的值
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


# 将输入的图像转换为 PIL.Image.Image 格式，并且如果需要，则重新缩放并将通道维度移到最后一个维度
def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "torch.Tensor", "tf.Tensor", "jnp.ndarray"],
    do_rescale: Optional[bool] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> "PIL.Image.Image":
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor` or `tf.Tensor`):
            The image to convert to the `PIL.Image` format.
        do_rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type and casting to `int` would result in a loss of precision,
            and `False` otherwise.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `PIL.Image.Image`: The converted image.
    """
    # 确保所需的后端已加载
    requires_backends(to_pil_image, ["vision"])

    if isinstance(image, PIL.Image.Image):
        return image

    # 将所有张量转换为 numpy 数组，以便转换为 PIL 图像
    if is_torch_tensor(image) or is_tf_tensor(image):
        image = image.numpy()
    elif is_jax_tensor(image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        # 抛出异常，因为不支持的输入图像类型
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # 如果通道维度已经移动到第一维度，我们将其放回到最后一个维度
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # 如果只有一个通道，我们压缩它，因为 PIL 不能处理非压缩的单通道图像
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image
    # 如果需要将图像转换为 PIL.Image 格式，确保其像素值在 0 到 255 之间。
    do_rescale = _rescale_for_pil_conversion(image) if do_rescale is None else do_rescale
    # 如果需要进行像素值的重新缩放，则调用 rescale 函数将图像像素值缩放到 0 到 255 的范围内。
    if do_rescale:
        image = rescale(image, 255)
    # 将图像的数据类型转换为 np.uint8，确保图像的像素值范围在 0 到 255 之间。
    image = image.astype(np.uint8)
    # 根据图像的 numpy 数组创建一个 PIL.Image 对象，并返回该对象。
    return PIL.Image.fromarray(image)
# 导入必要的库和模块
# Logic adapted from torchvision resizing logic: https://github.com/pytorch/vision/blob/511924c1ced4ce0461197e5caa64ce5b9e558aab/torchvision/transforms/functional.py#L366
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    default_to_square: bool = True,
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or List[int] or Tuple[int]):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be matched to
            this.

            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a square
            (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater
            than `max_size` after being resized according to `size`, then the image is resized again so that the longer
            edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter
            than `size`. Only used if `default_to_square` is `False`.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """

    # 如果 `size` 是一个元组或列表
    if isinstance(size, (tuple, list)):
        # 如果 `size` 的长度为2，直接返回元组形式的大小
        if len(size) == 2:
            return tuple(size)
        # 如果 `size` 的长度为1，执行和整数大小相同的逻辑
        elif len(size) == 1:
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    # 如果默认需要输出为正方形
    if default_to_square:
        return (size, size)

    # 获取输入图像的高度和宽度
    height, width = get_image_size(input_image, input_data_format)
    # 确定较短和较长的边
    short, long = (width, height) if width <= height else (height, width)
    # 请求的新的较短边的大小
    requested_new_short = size
    # 从请求中获取新的短边和长边尺寸，计算新的长边尺寸为请求的新短边尺寸乘以长边与短边的比例
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    # 如果设置了最大尺寸限制
    if max_size is not None:
        # 如果最大尺寸小于或等于请求的新短边尺寸，抛出值错误异常
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        # 如果新的长边超过了最大尺寸，调整新的短边和长边的尺寸比例，并将长边限制为最大尺寸
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    # 根据宽度和高度的比较，返回调整后的长短边尺寸元组
    return (new_long, new_short) if width <= height else (new_short, new_long)
    """
    使用 PIL 库将 `image` 调整大小为 `size` 指定的尺寸。

    Args:
        image (`np.ndarray`):
            要调整大小的图像。
        size (`Tuple[int, int]`):
            用于调整图像大小的尺寸。
        resample (`int`, *optional*, 默认为 `PILImageResampling.BILINEAR`):
            用于重采样的滤波器。
        reducing_gap (`int`, *optional*):
            通过两步骤优化图像调整大小。`reducing_gap` 越大，结果越接近公平重采样。详细信息请参考 Pillow 文档。
        data_format (`ChannelDimension`, *optional*):
            输出图像的通道维度格式。如果未设置，将从输入中推断格式。
        return_numpy (`bool`, *optional*, 默认为 `True`):
            是否将调整大小后的图像作为 numpy 数组返回。如果为 False，则返回 `PIL.Image.Image` 对象。
        input_data_format (`ChannelDimension`, *optional*):
            输入图像的通道维度格式。如果未设置，将从输入中推断格式。

    Returns:
        `np.ndarray`: 调整大小后的图像。
    """
    requires_backends(resize, ["vision"])

    # 如果未指定 resample 方法，则默认使用 BILINEAR 方法
    resample = resample if resample is not None else PILImageResampling.BILINEAR

    # 检查 size 是否包含两个元素，否则抛出 ValueError
    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # 对于所有转换，我们希望保持与输入图像相同的数据格式，除非另有指定。
    # PIL 调整大小后的图像始终将通道放在最后，因此首先找到输入格式。
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # 为了保持与以前图像特征提取器中所做的调整大小的向后兼容性，我们使用 Pillow 库调整大小图像，然后转换回 numpy 数组。
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        # 如果输入图像不是 PIL.Image.Image 对象，则进行相应的转换
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    
    # 提取出 size 的高度和宽度
    height, width = size
    # PIL 图像的大小顺序为 (宽度, 高度)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    # 如果需要返回一个 NumPy 数组，则将 resized_image 转换为 NumPy 数组类型
    resized_image = np.array(resized_image)
    # 如果输入图像的通道维度为 1，在转换为 PIL 图像时会丢失通道维度，因此需要在必要时添加回来
    resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
    # 在从 PIL 图像转换后，图像始终处于通道最后的格式
    resized_image = to_channel_dimension_format(
        resized_image, data_format, input_channel_dim=ChannelDimension.LAST
    )
    # 如果在转换为 PIL 图像之前对图像进行了 [0, 255] 范围内的重新缩放，则需要将其重新缩放回原始范围
    resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    # 返回处理后的图像
    return resized_image
# 定义函数 `center_crop`，用于对图像进行中心裁剪操作，返回裁剪后的图像
def center_crop(
    image: np.ndarray,
    size: Tuple[int, int],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).

    Args:
        image (`np.ndarray`):
            The input image to be cropped.
        size (`Tuple[int, int]`):
            The desired output size after cropping, specified as (height, width).
        data_format (`Union[str, ChannelDimension]`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`Union[str, ChannelDimension]`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
        return_numpy (`bool`, *optional*):
            Deprecated parameter. If provided, this should be set to `True`.

    Returns:
        `np.ndarray`: The cropped image of the specified `size`.
    """
    """
    Args:
        image (`np.ndarray`):
            The image to crop.
        size (`Tuple[int, int]`):
            The target size for the cropped image.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        return_numpy (`bool`, *optional*):
            Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the
            previous ImageFeatureExtractionMixin method.
                - Unset: will return the same type as the input image.
                - `True`: will return a numpy array.
                - `False`: will return a `PIL.Image.Image` object.
    Returns:
        `np.ndarray`: The cropped image.
    """
    requires_backends(center_crop, ["vision"])

    # Warn about deprecation of `return_numpy` parameter
    if return_numpy is not None:
        warnings.warn("return_numpy is deprecated and will be removed in v.4.33", FutureWarning)

    # Determine whether to return numpy array based on `return_numpy` parameter
    return_numpy = True if return_numpy is None else return_numpy

    # Validate input image type
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    # Validate size parameter
    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    # Determine input data format if not explicitly provided
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # Convert image to (C, H, W) format if necessary
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    # Get original image dimensions in channels-first format
    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # Calculate top-left corner coordinates of the crop area
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if the calculated crop area is within image boundaries
    # 如果裁剪区域在图片边界内，则直接裁剪
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        # 根据给定的裁剪区域对图像进行裁剪
        image = image[..., top:bottom, left:right]
        # 调整图像的通道维度格式为指定的输出格式，通道维度置于最前面
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        # 返回裁剪后的图像
        return image

    # 否则，如果图像太小，需要进行填充处理
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    # 构建新图像的形状，保留除了最后两个维度外的所有维度，并添加新的高度和宽度维度
    new_shape = image.shape[:-2] + (new_height, new_width)
    # 创建与原图像相同形状的全零数组作为新图像
    new_image = np.zeros_like(image, shape=new_shape)

    # 计算需要填充的边界
    top_pad = (new_height - orig_height) // 2
    bottom_pad = top_pad + orig_height
    left_pad = (new_width - orig_width) // 2
    right_pad = left_pad + orig_width
    # 在新图像的指定位置进行填充
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    # 更新裁剪区域的边界位置
    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    # 根据更新后的裁剪边界对新图像进行进一步裁剪
    new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
    # 调整图像的通道维度格式为指定的输出格式，通道维度置于最前面
    new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

    # 如果不需要返回 NumPy 数组，则转换成 PIL 图像格式
    if not return_numpy:
        new_image = to_pil_image(new_image)

    # 返回处理后的图像
    return new_image
# 将中心格式的边界框转换为角点格式的边界框（使用 PyTorch 张量）
def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    # 从中心格式的边界框张量中解绑出中心坐标和宽度、高度信息
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    # 计算角点格式的边界框张量：左上角 x、左上角 y、右下角 x、右下角 y
    bbox_corners = torch.stack(
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


# 将中心格式的边界框转换为角点格式的边界框（使用 NumPy 数组）
def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
    # 从中心格式的边界框数组中解绑出中心坐标和宽度、高度信息
    center_x, center_y, width, height = bboxes_center.T
    # 计算角点格式的边界框数组：左上角 x、左上角 y、右下角 x、右下角 y
    bboxes_corners = np.stack(
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners


# 将中心格式的边界框转换为角点格式的边界框（使用 TensorFlow 张量）
def _center_to_corners_format_tf(bboxes_center: "tf.Tensor") -> "tf.Tensor":
    # 从中心格式的边界框张量中解绑出中心坐标和宽度、高度信息
    center_x, center_y, width, height = tf.unstack(bboxes_center, axis=-1)
    # 计算角点格式的边界框张量：左上角 x、左上角 y、右下角 x、右下角 y
    bboxes_corners = tf.stack(
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners


# 以下两个函数灵感来自 https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
# 将边界框从中心格式转换为角点格式的统一接口函数
def center_to_corners_format(bboxes_center: TensorType) -> TensorType:
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coodinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    # 根据输入类型选择对应的转换函数，用于模型前向传递时的边界框格式转换，尽可能不转换为 NumPy 数组
    if is_torch_tensor(bboxes_center):  # 如果是 PyTorch 张量
        return _center_to_corners_format_torch(bboxes_center)
    elif isinstance(bboxes_center, np.ndarray):  # 如果是 NumPy 数组
        return _center_to_corners_format_numpy(bboxes_center)
    elif is_tf_tensor(bboxes_center):  # 如果是 TensorFlow 张量
        return _center_to_corners_format_tf(bboxes_center)

    # 如果输入类型不支持，则抛出异常
    raise ValueError(f"Unsupported input type {type(bboxes_center)}")


# 将角点格式的边界框转换为中心格式（使用 PyTorch 张量）
def _corners_to_center_format_torch(bboxes_corners: "torch.Tensor") -> "torch.Tensor":
    # 从角点格式的边界框张量中解绑出左上角和右下角坐标信息
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.unbind(-1)
    # 计算中心格式的边界框张量：中心 x、中心 y、宽度、高度
    b = [
        (top_left_x + bottom_right_x) / 2,  # 中心 x
        (top_left_y + bottom_right_y) / 2,  # 中心 y
        (bottom_right_x - top_left_x),      # 宽度
        (bottom_right_y - top_left_y),      # 高度
    ]
    return torch.stack(b, dim=-1)


# 将角点格式的边界框转换为中心格式（使用 NumPy 数组）
def _corners_to_center_format_numpy(bboxes_corners: np.ndarray) -> np.ndarray:
    # 从角点格式的边界框数组中解绑出左上角和右下角坐标信息
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
    # 创建一个包含边界框中心坐标和宽高的数组
    bboxes_center = np.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # 计算边界框的中心 x 坐标
            (top_left_y + bottom_right_y) / 2,  # 计算边界框的中心 y 坐标
            (bottom_right_x - top_left_x),      # 计算边界框的宽度
            (bottom_right_y - top_left_y),      # 计算边界框的高度
        ],
        axis=-1,  # 沿着最后一个轴（即最内层）堆叠数组
    )
    # 返回包含所有边界框中心和尺寸信息的数组
    return bboxes_center
def _corners_to_center_format_tf(bboxes_corners: "tf.Tensor") -> "tf.Tensor":
    """
    Converts bounding boxes from corners format to center format using TensorFlow operations.

    Args:
        bboxes_corners (tf.Tensor): Tensor containing bounding box coordinates in corners format
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    Returns:
        tf.Tensor: Tensor containing bounding box coordinates in center format
            (center_x, center_y, width, height)
    """
    # Unstack the input tensor along the last axis to get individual coordinates
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = tf.unstack(bboxes_corners, axis=-1)
    # Compute center coordinates, width, and height using TensorFlow operations
    bboxes_center = tf.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # center x
            (top_left_y + bottom_right_y) / 2,  # center y
            (bottom_right_x - top_left_x),      # width
            (bottom_right_y - top_left_y),      # height
        ],
        axis=-1,
    )
    return bboxes_center


def corners_to_center_format(bboxes_corners: TensorType) -> TensorType:
    """
    Converts bounding boxes from corners format to center format.

    Args:
        bboxes_corners (TensorType): Tensor or array containing bounding box coordinates in corners format
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    Returns:
        TensorType: Tensor or array containing bounding box coordinates in center format
            (center_x, center_y, width, height)

    Raises:
        ValueError: If the input type is unsupported
    """
    # Check the type of input and call the respective conversion function
    if is_torch_tensor(bboxes_corners):
        return _corners_to_center_format_torch(bboxes_corners)
    elif isinstance(bboxes_corners, np.ndarray):
        return _corners_to_center_format_numpy(bboxes_corners)
    elif is_tf_tensor(bboxes_corners):
        return _corners_to_center_format_tf(bboxes_corners)

    # Raise an error if the input type is not recognized
    raise ValueError(f"Unsupported input type {type(bboxes_corners)}")


# 2 functions below copied from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# Copyright (c) 2018, Alexander Kirillov
# All rights reserved.
def rgb_to_id(color):
    """
    Converts RGB color to unique ID.

    Args:
        color (np.ndarray or list): RGB color values

    Returns:
        int: Unique ID corresponding to the RGB color
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id_to_rgb(id_map):
    """
    Converts unique ID to RGB color.

    Args:
        id_map (np.ndarray or int): Unique ID or array of IDs

    Returns:
        np.ndarray or list: RGB color corresponding to the unique ID or array of RGB colors
    """
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


class PaddingMode(ExplicitEnum):
    """
    Enum class for the different padding modes to use when padding images.
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    SYMMETRIC = "symmetric"


def pad(
    image: np.ndarray,
    padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_values: Union[float, Iterable[float]] = 0.0,
    data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """
    Pads an image array according to specified parameters.

    Args:
        image (np.ndarray): Image array to be padded.
        padding (int or Tuple[int, int] or Iterable[Tuple[int, int]]): Padding size or sizes in each dimension.
        mode (PaddingMode, optional): Padding mode, defaults to PaddingMode.CONSTANT.
        constant_values (float or Iterable[float], optional): Constant value(s) to pad with, defaults to 0.0.
        data_format (str or ChannelDimension, optional): Data format of the image array, defaults to None.

    Returns:
        np.ndarray: Padded image array.
    """
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
# 使用 numpy 数组作为参数的函数定义，该函数用于对图像进行填充操作。
def pad_image(image: np.ndarray,
              padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
              mode: PaddingMode,
              constant_values: Optional[Union[float, Iterable[float]]] = None,
              data_format: Optional[Union[str, ChannelDimension]] = None,
              input_data_format: Optional[Union[str, ChannelDimension]] = None) -> np.ndarray:
    """
    Pads the `image` with the specified (height, width) `padding` and `mode`.

    Args:
        image (`np.ndarray`):
            The image to pad.
        padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
            Padding to apply to the edges of the height, width axes. Can be one of three formats:
            - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
            - `((before, after),)` yields same before and after pad for height and width.
            - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
        mode (`PaddingMode`):
            The padding mode to use. Can be one of:
                - `"constant"`: pads with a constant value.
                - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                  vector along each axis.
                - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
        constant_values (`float` or `Iterable[float]`, *optional*):
            The value to use for the padding if `mode` is `"constant"`.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.

    Returns:
        `np.ndarray`: The padded image.

    """
    # 如果未指定输入数据的通道格式，使用推断的通道格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    def _expand_for_data_format(values):
        """
        Convert values to be in the format expected by np.pad based on the data format.
        """
        # 如果values是整数或浮点数，将其转换为二维元组格式
        if isinstance(values, (int, float)):
            values = ((values, values), (values, values))
        # 如果values是长度为1的元组，将其转换为二维元组格式
        elif isinstance(values, tuple) and len(values) == 1:
            values = ((values[0], values[0]), (values[0], values[0]))
        # 如果values是长度为2的元组且第一个元素是整数，保持不变
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
            values = (values, values)
        # 如果values是长度为2的元组且第一个元素是元组，保持不变
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
            values = values
        else:
            # 如果values不符合以上格式，抛出异常
            raise ValueError(f"Unsupported format: {values}")

        # 根据输入数据格式选择是否在通道维度前面添加0
        values = ((0, 0), *values) if input_data_format == ChannelDimension.FIRST else (*values, (0, 0))

        # 如果图像维度为4，则在前面添加0作为批量维度
        values = (0, *values) if image.ndim == 4 else values
        return values

    # 根据数据格式扩展填充参数
    padding = _expand_for_data_format(padding)

    # 根据填充模式进行图像填充
    if mode == PaddingMode.CONSTANT:
        # 根据数据格式扩展常数填充值
        constant_values = _expand_for_data_format(constant_values)
        # 使用常数填充模式填充图像
        image = np.pad(image, padding, mode="constant", constant_values=constant_values)
    elif mode == PaddingMode.REFLECT:
        # 使用反射模式填充图像
        image = np.pad(image, padding, mode="reflect")
    elif mode == PaddingMode.REPLICATE:
        # 使用复制模式填充图像
        image = np.pad(image, padding, mode="edge")
    elif mode == PaddingMode.SYMMETRIC:
        # 使用对称模式填充图像
        image = np.pad(image, padding, mode="symmetric")
    else:
        # 如果填充模式无效，抛出异常
        raise ValueError(f"Invalid padding mode: {mode}")

    # 如果数据格式不为空，将图像转换为通道维度格式
    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image
# TODO (Amy): Accept 1/3/4 channel numpy array as input and return np.array as default
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.

    Args:
        image (Image):
            The image to convert.
    """
    # 确保当前函数所需的视觉后端已加载
    requires_backends(convert_to_rgb, ["vision"])

    # 如果传入的图像不是 PIL.Image.Image 类型，则直接返回
    if not isinstance(image, PIL.Image.Image):
        return image

    # 将图像转换为 RGB 格式
    image = image.convert("RGB")
    return image


def flip_channel_order(
    image: np.ndarray,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Flips the channel order of the image.

    If the image is in RGB format, it will be converted to BGR and vice versa.

    Args:
        image (`np.ndarray`):
            The image to flip.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
    """
    # 推断输入图像的通道维度格式，如果未指定则使用推断的格式
    input_data_format = infer_channel_dimension_format(image) if input_data_format is None else input_data_format

    # 根据输入图像的通道维度格式执行通道顺序翻转操作
    if input_data_format == ChannelDimension.LAST:
        image = image[..., ::-1]  # BGR 到 RGB 或 RGB 到 BGR 的转换
    elif input_data_format == ChannelDimension.FIRST:
        image = image[::-1, ...]  # BGR 到 RGB 或 RGB 到 BGR 的转换
    else:
        raise ValueError(f"Unsupported channel dimension: {input_data_format}")

    # 如果指定了输出图像的通道维度格式，则将图像转换为该格式
    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    return image
```