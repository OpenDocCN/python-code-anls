# `.\models\vitmatte\image_processing_vitmatte.py`

```
# 导入所需模块和类
from typing import List, Optional, Union

import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵操作

# 导入所需的图像处理工具和实用函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import pad, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入图像处理时所需的标准均值
    IMAGENET_STANDARD_STD,   # 导入图像处理时所需的标准标准差
    ChannelDimension,        # 导入通道维度枚举类
    ImageInput,              # 导入图像输入类
    get_image_size,          # 导入获取图像尺寸的函数
    infer_channel_dimension_format,  # 推断通道维度格式的函数
    is_scaled_image,         # 判断图像是否为缩放图像的函数
    make_list_of_images,     # 将图像处理为图像列表的函数
    to_numpy_array,          # 将输入转换为 NumPy 数组的函数
    valid_images,            # 验证图像有效性的函数
    validate_kwargs,         # 验证关键字参数的函数
    validate_preprocess_arguments,  # 验证预处理参数的函数
)
from ...utils import TensorType, logging  # 导入张量类型和日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象
    """
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

    # 定义模型输入的名称列表，只包含一个元素 "pixel_values"
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        size_divisibility: int = 32,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化类的属性，设置各个参数的默认值或者根据传入的参数进行设置
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_pad = do_pad
        self.rescale_factor = rescale_factor
        # 如果传入的 image_mean 参数不为 None，则使用传入的值；否则使用预设的 IMAGENET_STANDARD_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 如果传入的 image_std 参数不为 None，则使用传入的值；否则使用预设的 IMAGENET_STANDARD_STD
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.size_divisibility = size_divisibility
        # 设置有效的处理器键名列表，用于后续数据处理
        self._valid_processor_keys = [
            "images",
            "trimaps",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "size_divisibility",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
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
        # 推断输入图像的通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 获取图像的高度和宽度
        height, width = get_image_size(image, input_data_format)

        # 如果图像的高度或宽度不是size_divisibility的整数倍，则进行填充
        if height % size_divisibility != 0 or width % size_divisibility != 0:
            pad_height = size_divisibility - height % size_divisibility
            pad_width = size_divisibility - width % size_divisibility
            padding = ((0, pad_height), (0, pad_width))
            # 对图像进行填充操作，保证其高度和宽度是size_divisibility的整数倍
            image = pad(image, padding=padding, data_format=data_format, input_data_format=input_data_format)

        # 如果指定了输出图像的通道维度格式，则将图像转换为该格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_data_format)

        # 返回填充或转换后的图像
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
    ):
        # 这部分的注释应该由你来完成，因为它们和上述的代码块并不相关。
```