# `.\models\vivit\image_processing_vivit.py`

```py
# 指定编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0
# 详细版权信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
"""Vivit 的图像处理类。"""

# 引入必要的类型声明
from typing import Dict, List, Optional, Union

# 引入 numpy 库并重命名为 np
import numpy as np

# 从 transformers.utils 中导入 is_vision_available 函数
from transformers.utils import is_vision_available

# 从 transformers.utils.generic 中导入 TensorType 类型
from transformers.utils.generic import TensorType

# 导入自定义的图像处理工具和相关函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    rescale,
    resize,
    to_channel_dimension_format,
)
# 导入与图像处理相关的工具函数和常量
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入 logging 模块
from ...utils import logging

# 如果 vision 可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取 logger 对象
logger = logging.get_logger(__name__)


def make_batched(videos) -> List[List[ImageInput]]:
    """将视频列表批量化为 Vivit 需要的格式。

    Args:
        videos: 输入的视频数据，可以是单个视频或嵌套列表/元组的视频集合。

    Returns:
        List[List[ImageInput]]: 批量化后的视频列表，每个元素为一个视频帧列表。
    
    Raises:
        ValueError: 如果无法从给定的视频数据创建批量化视频。
    """
    # 如果 videos 是嵌套列表或元组，并且第一个元素是嵌套列表或元组，并且第一个视频帧是有效图像
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    # 如果 videos 是列表或元组，并且第一个元素是有效图像
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]

    # 如果 videos 是有效图像
    elif is_valid_image(videos):
        return [[videos]]

    # 如果无法从 videos 创建批量化视频，抛出 ValueError 异常
    raise ValueError(f"Could not make batched video from {videos}")


class VivitImageProcessor(BaseImageProcessor):
    r"""
    构建 Vivit 图像处理器。

    继承自 BaseImageProcessor 类。
    """
    def __init__(self):
        """初始化 Vivit 图像处理器。"""
        super().__init__()
    # 定义函数参数和默认值，用于图像预处理
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的高度和宽度尺寸到指定的 `size`。可以在 `preprocess` 方法中的 `do_resize` 参数中被覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            调整后的输出图像尺寸。图像的最短边将调整为 `size["shortest_edge"]`，同时保持原始图像的纵横比。可以在 `preprocess` 方法中的 `size` 参数中被覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            调整图像尺寸时使用的重采样滤波器。可以在 `preprocess` 方法中的 `resample` 参数中被覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪到指定的 `crop_size`。可以在 `preprocess` 方法中的 `do_center_crop` 参数中被覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            应用中心裁剪后的图像尺寸。可以在 `preprocess` 方法中的 `crop_size` 参数中被覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的缩放因子 `rescale_factor` 进行图像缩放。可以在 `preprocess` 方法中的 `do_rescale` 参数中被覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/127.5`):
            如果进行图像缩放，定义要使用的缩放因子。可以在 `preprocess` 方法中的 `rescale_factor` 参数中被覆盖。
        offset (`bool`, *optional*, defaults to `True`):
            是否在正负方向同时进行图像缩放。可以在 `preprocess` 方法中的 `offset` 参数中被覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以在 `preprocess` 方法中的 `do_normalize` 参数中被覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果归一化图像，定义要使用的均值。这是一个浮点数或长度等于图像通道数的浮点数列表。可以在 `preprocess` 方法中的 `image_mean` 参数中被覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果归一化图像，定义要使用的标准差。这是一个浮点数或长度等于图像通道数的浮点数列表。可以在 `preprocess` 方法中的 `image_std` 参数中被覆盖。
    """
    
    model_input_names = ["pixel_values"]
    # 初始化函数，用于设置图像预处理的各项参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像尺寸调整的标志
        size: Dict[str, int] = None,  # 图像尺寸的字典，包含最短边和可能的其他尺寸参数
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整大小时的重采样方法
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 裁剪后的图像尺寸的字典，包含高度和宽度
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志
        rescale_factor: Union[int, float] = 1 / 127.5,  # 图像像素值缩放的因子
        offset: bool = True,  # 是否进行图像像素值偏移的标志
        do_normalize: bool = True,  # 是否进行图像像素值标准化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像像素值的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像像素值的标准差
        **kwargs,  # 其他参数，以字典形式传入
    ) -> None:
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)
        
        # 如果 size 参数为 None，则设为默认值 {"shortest_edge": 256}
        size = size if size is not None else {"shortest_edge": 256}
        # 根据参数获取最终确定的图像尺寸字典，确保不是正方形
        size = get_size_dict(size, default_to_square=False)
        
        # 如果 crop_size 参数为 None，则设为默认值 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据参数获取最终确定的裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 将各参数值赋给对象的属性
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.offset = offset
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        
        # 图像处理器对象支持的键列表
        self._valid_processor_keys = [
            "videos",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "offset",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    # 重新调整图像大小的函数，基于输入参数对图像进行变换

    # size参数表示输出图像的尺寸，根据get_size_dict函数获取确切的尺寸字典
    size = get_size_dict(size, default_to_square=False)

    # 如果size字典中包含"shortest_edge"键，根据最短边长度调整输出图像尺寸
    if "shortest_edge" in size:
        # 调用get_resize_output_image_size函数计算调整后的图像尺寸
        output_size = get_resize_output_image_size(
            image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
        )
    # 如果size字典中同时包含"height"和"width"键，直接使用指定的高度和宽度
    elif "height" in size and "width" in size:
        output_size = (size["height"], size["width"])
    # 如果size字典中的键不符合要求，抛出数值错误异常
    else:
        raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")

    # 调用resize函数对图像进行实际的大小调整操作，返回调整后的图像
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
        # 调用rescale函数，对图像进行重新缩放处理
        rescaled_image = rescale(
            image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

        # 如果offset为True，则对重新缩放后的图像进行偏移处理
        if offset:
            rescaled_image = rescaled_image - 1

        # 返回经过缩放和可能偏移处理后的图像
        return rescaled_image

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
        offset: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        # 验证预处理参数的有效性，确保所有参数都被正确设置
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

        if offset and not do_rescale:
            # 如果设置了 offset 但未设置 do_rescale，则抛出数值错误异常
            raise ValueError("For offset, do_rescale must also be set to True.")

        # 将输入的图像转换为 numpy 数组
        image = to_numpy_array(image)

        if is_scaled_image(image) and do_rescale:
            # 如果图像已经被缩放，并且需要进行重新缩放，则发出警告
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # 推断输入数据的通道维度格式
            input_data_format = infer_channel_dimension_format(image)

        if do_resize:
            # 如果需要进行 resize 操作，则调用 resize 方法进行处理
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            # 如果需要进行中心裁剪，则调用 center_crop 方法进行处理
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            # 如果需要进行缩放操作，则调用 rescale 方法进行处理
            image = self.rescale(image=image, scale=rescale_factor, offset=offset, input_data_format=input_data_format)

        if do_normalize:
            # 如果需要进行归一化，则调用 normalize 方法进行处理
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # 将图像数据转换为指定的通道维度格式
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def preprocess(
        self,
        videos: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        offset: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```