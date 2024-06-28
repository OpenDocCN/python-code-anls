# `.\models\pvt\image_processing_pvt.py`

```py
# 导入所需的模块和类
from typing import Dict, List, Optional, Union  # 导入类型提示模块

import numpy as np  # 导入NumPy库

# 导入图像处理相关的工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,  # 导入常量：ImageNet图像的默认均值
    IMAGENET_DEFAULT_STD,   # 导入常量：ImageNet图像的默认标准差
    ChannelDimension,       # 导入枚举类型：通道维度
    ImageInput,             # 导入类型别名：图像输入
    PILImageResampling,     # 导入枚举类型：PIL图像的重采样方法
    infer_channel_dimension_format,  # 导入函数：推断通道维度格式
    is_scaled_image,        # 导入函数：判断图像是否被缩放过
    make_list_of_images,    # 导入函数：创建图像列表
    to_numpy_array,         # 导入函数：将图像转换为NumPy数组
    valid_images,           # 导入函数：验证图像的有效性
    validate_kwargs,        # 导入函数：验证关键字参数
    validate_preprocess_arguments,  # 导入函数：验证预处理参数
)
from ...utils import TensorType, logging  # 导入类型别名和日志记录模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
    """
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
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    # 定义一个图像处理器类，继承自父类，用于处理图像的预处理操作
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法，传入所有关键字参数
        super().__init__(**kwargs)
        # 确定图像大小，若未指定则使用默认大小 {"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 调用辅助函数，将 size 转换为规范化的尺寸字典
        size = get_size_dict(size)
        # 是否进行图像大小调整的标志位
        self.do_resize = do_resize
        # 是否进行图像尺度缩放的标志位
        self.do_rescale = do_rescale
        # 是否进行图像归一化的标志位
        self.do_normalize = do_normalize
        # 存储图像大小的字典
        self.size = size
        # 图像调整使用的重采样方法
        self.resample = resample
        # 图像尺度缩放的因子
        self.rescale_factor = rescale_factor
        # 图像均值，若未指定则使用默认的 IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 图像标准差，若未指定则使用默认的 IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 存储有效的处理器关键字列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    
    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制而来的方法
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
        size = get_size_dict(size)  # 获取调整后的尺寸字典
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # 设置输出图像的高度和宽度
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
    ):
```