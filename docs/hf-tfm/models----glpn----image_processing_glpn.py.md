# `.\models\glpn\image_processing_glpn.py`

```py
# 设置编码格式为utf-8
# 版权声明
# 根据Apache License, Version 2.0授权许可证，您不得使用此文件，除非您遵守许可证的规定。您可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发，并且没有任何形式的担保或条件，无论是明示或暗示的。
# 请查看许可证，了解特定语言规定的权限和限制。
"""GLPN的图像处理器类"""
# 导入必要的库
from typing import List, Optional, Union
import numpy as np
import PIL.Image
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging
# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义GLPNImageProcessor类，继承自BaseImageProcessor
class GLPNImageProcessor(BaseImageProcessor):
    r"""
    构造一个GLPN图像处理器。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）维度，将其舍入到最接近`size_divisor`的倍数。可以通过`preprocess`中的`do_resize`覆盖。
        size_divisor (`int`, *optional*, defaults to 32):
            当`do_resize`为`True`时，图像将被调整大小，使其高度和宽度舍入到最接近`size_divisor`的倍数。可以通过`preprocess`中的`size_divisor`覆盖。
        resample (`PIL.Image` resampling filter, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，则使用的重采样滤波器。可以通过`preprocess`中的`resample`覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否应用缩放因子（使像素值浮动在0.和1.之间）。可以通过`preprocess`中的`do_rescale`覆盖。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        resample=PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        **kwargs,
    ) -> None:
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size_divisor: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(self, image: np.ndarray, size_divisor: int, resample, data_format: ChannelDimension, input_data_format: Union[str, ChannelDimension]) -> np.ndarray:
        """
        Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.

        If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).

        Args:
            image (`np.ndarray`):
                The image to resize.
            size_divisor (`int`):
                The image is resized so its height and width are rounded down to the closest multiple of `size_divisor`.
            resample:
                `PIL.Image` resampling filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If `None`, the channel dimension format of the input
                image is used. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not set, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        # calculate the height and width of the input image
        height, width = get_image_size(image, channel_dim=input_data_format)
        # Rounds the height and width down to the closest multiple of size_divisor
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        # resize the image using the specified parameters
        image = resize(
            image,
            (new_h, new_w),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return image

    def preprocess(
        self,
        images: Union["PIL.Image.Image", TensorType, List["PIL.Image.Image"], List[TensorType]],
        do_resize: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```