# `.\transformers\models\swin2sr\image_processing_swin2sr.py`

```py
# 导入必要的模块和类
from typing import Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import get_image_size, pad, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging

logger = logging.get_logger(__name__)

# 定义 Swin2SRImageProcessor 类，继承自 BaseImageProcessor
class Swin2SRImageProcessor(BaseImageProcessor):
    r"""
    构造一个 Swin2SR 图像处理器类。

    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否对图像进行指定的比例缩放。可以在 `preprocess` 方法中通过 `do_rescale` 参数进行修改。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果对图像进行缩放，则使用的缩放因子。可以在 `preprocess` 方法中通过 `rescale_factor` 参数进行修改。
    """

    # 输出模型的输入名称
    model_input_names = ["pixel_values"]

    # 定义初始化方法
    def __init__(
        self,
        do_rescale: bool = True,  # 是否对图像进行指定的比例缩放
        rescale_factor: Union[int, float] = 1 / 255,  # 缩放因子，默认为 1/255
        do_pad: bool = True,  # 是否进行填充
        pad_size: int = 8,  # 填充大小，默认为 8
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置属性
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size = pad_size

    # 定义填充方法
    def pad(
        self,
        image: np.ndarray,
        size: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        ...

以上是给定代码的注释。
    ):
        """
        Pad an image to make the height and width divisible by `size`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`int`):
                The size to make the height and width divisible by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The padded image.
        """
        # 获取输入图片的高度和宽度
        old_height, old_width = get_image_size(image, input_data_format)
        # 计算需要填充的高度和宽度
        pad_height = (old_height // size + 1) * size - old_height
        pad_width = (old_width // size + 1) * size - old_width

        # 调用pad函数进行填充操作
        return pad(
            image,
            ((0, pad_height), (0, pad_width)),
            mode="symmetric",
            data_format=data_format,
            input_data_format=input_data_format,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```