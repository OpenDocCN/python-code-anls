# `.\models\efficientnet\image_processing_efficientnet.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 使用此文件，除非符合许可证下的条件，否则无法使用本文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不提供保证或条件，软件按"原样"分发
# 无论是明示的还是隐含的，都没有任何保证或条件
# 详见许可证，以了解特定语言的权限和限制
"""Image processor class for EfficientNet."""
# 导入所需的模块和类型注解
from typing import Dict, List, Optional, Union

import numpy as np

# 导入自定义的图像处理工具模块和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
from ...image_transforms import rescale, resize, to_channel_dimension_format
# 导入图像处理用到的工具函数和常量
from ...image_utils import (
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
)
# 导入通用工具函数和类型判断函数
from ...utils import TensorType, is_vision_available, logging

# 如果视觉处理模块可用，导入相应模块
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 EfficientNet 的图像处理类，继承自 BaseImageProcessor
class EfficientNetImageProcessor(BaseImageProcessor):
    r"""
    Constructs a EfficientNet image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            # 是否将图像的（高度，宽度）尺寸调整为指定的`size`。可以被`preprocess`中的`do_resize`覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 346, "width": 346}`):
            # 在`resize`后的图像尺寸。可以被`preprocess`中的`size`覆盖。
        resample (`PILImageResampling` filter, *optional*, defaults to 0):
            # 在调整图像大小时要使用的重采样滤镜。可以被`preprocess`中的`resample`覆盖。
        do_center_crop (`bool`, *optional*, defaults to `False`):
            # 是否中心裁剪图像。如果输入尺寸在任何边上小于`crop_size`，则用0填充图像，然后进行中心裁剪。可以被`preprocess`中的`do_center_crop`覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 289, "width": 289}`):
            # 应用中心裁剪时的期望输出尺寸。可以被`preprocess`中的`crop_size`覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            # 如果重新调整图像，则使用的比例因子。可以被`preprocess`方法中的`rescale_factor`参数覆盖。
        rescale_offset (`bool`, *optional*, defaults to `False`):
            # 是否将图像重新调整为[-scale_range，scale_range]，而不是[0，scale_range]。可以被`preprocess`方法中的`rescale_factor`参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            # 是否按照指定的比例`rescale_factor`重新调整图像。可以被`preprocess`方法中的`do_rescale`参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            # 是否对图像进行规范化。可以被`preprocess`方法中的`do_normalize`参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            # 在规范化图像时使用的均值。这是一个浮点数或长度等于图像通道数的浮点数列表。可以被`preprocess`方法中的`image_mean`参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            # 在规范化图像时使用的标准差。这是一个浮点数或长度等于图像通道数的浮点数列表。可以被`preprocess`方法中的`image_std`参数覆盖。
        include_top (`bool`, *optional*, defaults to `True`):
            # 是否再次调整图像。如果用于图像分类，则应设置为True。
    """

    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,  # 初始化函数，设置是否进行调整大小的标志，默认为True
        size: Dict[str, int] = None,  # 设置大小的字典，默认为None
        resample: PILImageResampling = PIL.Image.NEAREST,  # 设置重采样方法，默认为最近邻插值
        do_center_crop: bool = False,  # 设置是否进行中心裁剪的标志，默认为False
        crop_size: Dict[str, int] = None,  # 设置裁剪大小的字典，默认为None
        rescale_factor: Union[int, float] = 1 / 255,  # 设置缩放因子，默认为1/255
        rescale_offset: bool = False,  # 设置是否进行缩放偏移的标志，默认为False
        do_rescale: bool = True,  # 设置是否进行重新缩放的标志，默认为True
        do_normalize: bool = True,  # 设置是否进行归一化的标志，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为None
        include_top: bool = True,  # 设置是否包含顶部的标志，默认为True
        **kwargs,
    ) -> None:  # 函数返回None
        super().__init__(**kwargs)  # 调用父类的初始化函数
        size = size if size is not None else {"height": 346, "width": 346}  # 如果size不为None，则使用size，否则使用默认大小
        size = get_size_dict(size)  # 调用get_size_dict函数获取处理后的size字典
        crop_size = crop_size if crop_size is not None else {"height": 289, "width": 289}  # 如果crop_size不为None，则使用crop_size，否则使用默认大小
        crop_size = get_size_dict(crop_size, param_name="crop_size")  # 调用get_size_dict函数获取处理后的crop_size字典

        # 设置各项参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.rescale_offset = rescale_offset
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.include_top = include_top

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize with PILImageResampling.BILINEAR->PILImageResampling.NEAREST
    def resize(
        self,
        image: np.ndarray,  # 要调整大小的图像数组
        size: Dict[str, int],  # 调整的目标大小字典
        resample: PILImageResampling = PILImageResampling.NEAREST,  # 设置重采样方法，默认为最近邻插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,  # 其他参数
    ) -> np.ndarray:
        """
        将图像调整大小为`(size["height"], size["width"])`。

        Args:
            image (`np.ndarray`):
                要调整大小的图像。
            size (`Dict[str, int]`):
                以`{"height": int, "width": int}`格式指定输出图像的大小的字典。
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.NEAREST`):
                调整图像大小时要使用的`PILImageResampling`滤波器，例如`PILImageResampling.NEAREST`。
            data_format (`ChannelDimension` or `str`, *optional*):
                输出图像的通道维度格式。如果未设置，则使用输入图像的通道维度格式。可以是以下之一：
                - `"channels_first"`或`ChannelDimension.FIRST`：图像以(num_channels, height, width)格式。
                - `"channels_last"`或`ChannelDimension.LAST`：图像以(height, width, num_channels)格式。
                - `"none"`或`ChannelDimension.NONE`：图像以(height, width)格式。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。如果未设置，则从输入图像中推断出通道维度格式。可以是以下之一：
                - `"channels_first"`或`ChannelDimension.FIRST`：图像以(num_channels, height, width)格式。
                - `"channels_last"`或`ChannelDimension.LAST`：图像以(height, width, num_channels)格式。
                - `"none"`或`ChannelDimension.NONE`：图像以(height, width)格式。

        Returns:
            `np.ndarray`: 调整大小后的图像。
        """
        size = get_size_dict(size)
        # 检查字典中是否包含键`height`和`width`
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        # 调用`resize`函数，返回调整大小后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        offset: bool = True,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        """
        Rescale an image by a scale factor.

        如果 `offset` 是 `True`，那么图像的值将被 `scale` 重新调整，然后再加上1。 如果 `scale` 是 1/127.5，图像将在 [-1, 1] 之间重新调整。
            image = image * scale - 1

        如果 `offset` 是 `False`，并且 `scale` 是 1/255，那么图像将在 [0, 1] 之间重新调整。
            image = image * scale

        参数:
            image (`np.ndarray`):
                需要重新调整的图像.
            scale (`int` or `float`):
                应用于图像的缩放比例.
            offset (`bool`, *optional*):
                是否在正负方向上重新调整图像.
            data_format (`str` or `ChannelDimension`, *optional*):
                图像的通道维度格式. 如果未提供，将与输入图像相同.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式. 如果未提供，将被推断.
        """
        # 通过给定的参数重新调整图像
        rescaled_image = rescale(
            image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

        # 如果 offset 为 True，则对结果图像进行偏移
        if offset:
            rescaled_image = rescaled_image - 1

        # 返回重新调整后的图像
        return rescaled_image

    # 对图像进行预处理
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
        ):
```