# `.\models\tvp\image_processing_tvp.py`

```py
# coding=utf-8
# 版权 2023 年 Intel AIA 团队作者和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 Version 2.0 许可；除非符合许可，否则不得使用此文件。
# 您可以在以下网址获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可分发的软件是基于"原样"分发的，
# 没有任何形式的明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。

"""用于 TVP 的图像处理器类。"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# 从 image_processing_utils 中导入所需的模块和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 从 image_transforms 中导入所需的函数
from ...image_transforms import (
    PaddingMode,
    flip_channel_order,
    pad,
    resize,
    to_channel_dimension_format,
)
# 从 image_utils 中导入所需的常量和函数
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 从 utils 中导入所需的类型和函数
from ...utils import TensorType, is_vision_available, logging

# 如果视觉库可用，则导入 PIL
if is_vision_available():
    import PIL

# 获取 logger 对象
logger = logging.get_logger(__name__)


# 从 transformers.models.vivit.image_processing_vivit.make_batched 复制而来
def make_batched(videos) -> List[List[ImageInput]]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]
    elif is_valid_image(videos):
        return [[videos]]
    # 如果无法生成批处理视频，则引发 ValueError
    raise ValueError(f"Could not make batched video from {videos}")


def get_resize_output_image_size(
    input_image: np.ndarray,
    max_size: int = 448,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    # 获取输入图像的高度和宽度
    height, width = get_image_size(input_image, input_data_format)
    # 根据图像的长宽比例调整新的高度和宽度
    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = int(new_height * ratio)
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = int(new_width * ratio)
    size = (new_height, new_width)
    return size


class TvpImageProcessor(BaseImageProcessor):
    r"""
    构建一个 Tvp 图像处理器。

    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        pad_mode: PaddingMode = PaddingMode.CONSTANT,
        do_normalize: bool = True,
        do_flip_channel_order: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # 如果没有传入 size 参数，则设定默认值为 {"longest_edge": 448}
        size = size if size is not None else {"longest_edge": 448}
        # 如果没有传入 crop_size 参数，则设定默认值为 {"height": 448, "width": 448}
        crop_size = crop_size if crop_size is not None else {"height": 448, "width": 448}
        # 如果没有传入 pad_size 参数，则设定默认值为 {"height": 448, "width": 448}
        pad_size = pad_size if pad_size is not None else {"height": 448, "width": 448}

        # 初始化各个属性值
        self.do_resize = do_resize  # 是否进行 resize 操作的标志
        self.size = size  # 图像尺寸相关的设定
        self.do_center_crop = do_center_crop  # 是否进行中心裁剪的标志
        self.crop_size = crop_size  # 裁剪后的图像尺寸设定
        self.resample = resample  # resize 时使用的重采样方法
        self.do_rescale = do_rescale  # 是否进行图像像素值的重新缩放
        self.rescale_factor = rescale_factor  # 图像像素值缩放的比例因子
        self.do_pad = do_pad  # 是否进行图像的填充操作
        self.pad_size = pad_size  # 图像填充的目标尺寸设定
        self.constant_values = constant_values  # 图像填充时使用的常数填充值
        self.pad_mode = pad_mode  # 图像填充时使用的填充模式
        self.do_normalize = do_normalize  # 是否进行图像的归一化操作
        self.do_flip_channel_order = do_flip_channel_order  # 是否翻转图像通道顺序的标志
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 图像归一化的均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 图像归一化的标准差
        # 有效的处理器关键字列表，用于后续验证
        self._valid_processor_keys = [
            "videos",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_pad",
            "pad_size",
            "constant_values",
            "pad_mode",
            "do_normalize",
            "do_flip_channel_order",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"longest_edge": s}`, the output image will have its
                longest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Determine the actual size to resize the image to
        size = get_size_dict(size, default_to_square=False)

        # Check if both 'height' and 'width' are provided in the size dictionary
        if "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        # If only 'longest_edge' is provided, calculate the output size accordingly
        elif "longest_edge" in size:
            output_size = get_resize_output_image_size(image, size["longest_edge"], input_data_format)
        else:
            # Raise an error if neither 'height' and 'width' nor 'longest_edge' are specified
            raise ValueError(f"Size must have 'height' and 'width' or 'longest_edge' as keys. Got {size.keys()}")

        # Perform the resizing operation using specified parameters
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
        Pad an image with zeros to the given size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`)
                Size of the output image with pad.
            constant_values (`Union[float, Iterable[float]]`)
                The fill value to use when padding the image.
            pad_mode (`PaddingMode`)
                The pad mode, default to PaddingMode.CONSTANT
            data_format (`ChannelDimension` or `str`, *optional*)
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输入图像的高度和宽度
        height, width = get_image_size(image, channel_dim=input_data_format)
        
        # 获取要填充到的最大高度和宽度
        max_height = pad_size.get("height", height)
        max_width = pad_size.get("width", width)

        # 计算需要填充的右边和底部的像素数
        pad_right, pad_bottom = max_width - width, max_height - height
        
        # 如果计算出的填充量小于零，抛出值错误异常
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("The padding size must be greater than image size")

        # 构建填充元组，用于指定图像的填充方式
        padding = ((0, pad_bottom), (0, pad_right))
        
        # 调用填充函数，对图像进行填充
        padded_image = pad(
            image,
            padding,
            mode=pad_mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        # 返回填充后的图像
        return padded_image

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
        do_pad: bool = True,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = None,
        pad_mode: PaddingMode = None,
        do_normalize: bool = None,
        do_flip_channel_order: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        # 验证预处理参数的有效性
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=pad_size,  # 这里的 pad() 方法仅需要 pad_size 参数。
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # 所有的转换操作都期望输入为 numpy 数组
        image = to_numpy_array(image)

        if do_resize:
            # 若需要调整大小，则调用 resize 方法进行大小调整
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            # 若需要中心裁剪，则调用 center_crop 方法进行裁剪操作
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            # 若需要重新缩放，则调用 rescale 方法进行缩放操作
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            # 若需要归一化，则调用 normalize 方法进行归一化操作
            image = self.normalize(
                image=image.astype(np.float32), mean=image_mean, std=image_std, input_data_format=input_data_format
            )

        if do_pad:
            # 若需要填充，则调用 pad_image 方法进行填充操作
            image = self.pad_image(
                image=image,
                pad_size=pad_size,
                constant_values=constant_values,
                pad_mode=pad_mode,
                input_data_format=input_data_format,
            )

        # 预训练模型的检查点假设图像为 BGR 格式，而非 RGB 格式
        if do_flip_channel_order:
            # 若需要翻转通道顺序，则调用 flip_channel_order 方法进行翻转操作
            image = flip_channel_order(image=image, input_data_format=input_data_format)

        # 将图像转换为指定的通道维度格式
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image
```