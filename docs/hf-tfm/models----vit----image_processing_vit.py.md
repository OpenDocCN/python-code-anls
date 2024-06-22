# `.\transformers\models\vit\image_processing_vit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""ViT 的图像处理器类。"""

from typing import Dict, List, Optional, Union

import numpy as np

# 导入必要的模块和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
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
from ...utils import TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 ViTImageProcessor 类，继承自 BaseImageProcessor 类
class ViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
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
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        **kwargs,
    # 初始化函数，设置各种参数
    def __init__(self, size: Optional[Union[int, Tuple[int, int]]] = None, do_resize: bool = True, do_rescale: bool = False,
                 do_normalize: bool = False, resample: PILImageResampling = PILImageResampling.BILINEAR,
                 rescale_factor: float = 1.0, image_mean: Optional[Union[float, Tuple[float, float, float]]] = None,
                 image_std: Optional[Union[float, Tuple[float, float, float]]] = None, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 如果没有指定大小，则设置默认大小为 224x224
        size = size if size is not None else {"height": 224, "width": 224}
        # 将大小参数转换为字典格式
        size = get_size_dict(size)
        # 设置是否进行 resize 操作的标志
        self.do_resize = do_resize
        # 设置是否进行 rescale 操作的标志
        self.do_rescale = do_rescale
        # 设置是否进行 normalize 操作的标志
        self.do_normalize = do_normalize
        # 设置图片大小
        self.size = size
        # 设置重采样方法
        self.resample = resample
        # 设置缩放因子
        self.rescale_factor = rescale_factor
        # 设置图片均值，如果未指定则使用默认值
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 设置图片标准差，如果未指定则使用默认值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    # 图片 resize 函数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
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
        # Convert size to a standardized format
        size = get_size_dict(size)
        # Check if the required keys are present in the size dictionary
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # Extract the height and width values from the size dictionary
        output_size = (size["height"], size["width"])
        # Resize the image using the specified parameters
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```