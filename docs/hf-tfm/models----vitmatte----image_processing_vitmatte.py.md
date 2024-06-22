# `.\transformers\models\vitmatte\image_processing_vitmatte.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""Image processor class for ViTMatte.""" 的图像处理器类

# 导入必要的模块和类型提示
from typing import List, Optional, Union

import numpy as np

# 导入相关的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import pad, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
# 导入日志记录器
from ...utils import TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 VitMatteImageProcessor 类，继承自 BaseImageProcessor 类
class VitMatteImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViTMatte image processor.
    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to make the width and height divisible by `size_divisibility`. Can be overridden
            by the `do_pad` parameter in the `preprocess` method.
        size_divisibility (`int`, *optional*, defaults to 32):
            The width and height of the image will be padded to be divisible by this number.
    """

    model_input_names = ["pixel_values"]

    # 初始化方法，设置各种参数
    def __init__(
        self,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        do_pad: bool = True,
        size_divisibility: int = 32,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置是否进行缩放、归一化、填充的参数
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_pad = do_pad
        # 设置缩放因子、均值、标准差、填充因子的参数
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.size_divisibility = size_divisibility

    # 对图像进行填充的方法
    def pad_image(
        self,
        image: np.ndarray,
        size_divisibility: int = 32,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Args:
            image (`np.ndarray`):
                Image to pad.
            size_divisibility (`int`, *optional*, defaults to 32):
                The width and height of the image will be padded to be divisible by this number.
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
        # 如果输入数据格式未设置，则从输入图像中推断出通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 获取图像的高度和宽度
        height, width = get_image_size(image, input_data_format)

        # 如果高度或宽度不能被size_divisibility整除，则进行填充
        if height % size_divisibility != 0 or width % size_divisibility != 0:
            pad_height = size_divisibility - height % size_divisibility
            pad_width = size_divisibility - width % size_divisibility
            padding = ((0, pad_height), (0, pad_width))
            image = pad(image, padding=padding, data_format=data_format, input_data_format=input_data_format)

        # 如果数据格式不为空，则将图像转换为指定的通道维度格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_data_format)

        # 返回处理后的图像
        return image

    def preprocess(
        self,
        images: ImageInput,
        trimaps: ImageInput,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        size_divisibility: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```