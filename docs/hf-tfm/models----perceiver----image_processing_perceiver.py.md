# `.\transformers\models\perceiver\image_processing_perceiver.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
"""Perceiver 的图像处理器类。"""

# 导入所需模块和类型
from typing import Dict, List, Optional, Union
# 导入 NumPy 库并重命名为 np
import numpy as np

# 从相应位置导入图像处理工具和函数
# BaseImageProcessor 是一个抽象基类，提供了基本的图像处理功能
# BatchFeature 用于表示批量特征数据
# get_size_dict 用于获取尺寸信息的字典
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
# center_crop 用于中心裁剪图像
# resize 用于调整图像大小
# to_channel_dimension_format 用于将图像转换为指定通道维度格式
from ...image_transforms import center_crop, resize, to_channel_dimension_format
# 导入图像处理相关的实用函数和常量
# IMAGENET_DEFAULT_MEAN 和 IMAGENET_DEFAULT_STD 是图像处理中使用的默认均值和标准差
# ChannelDimension 是通道维度的枚举类型
# ImageInput 用于表示图像输入的数据类型
# PILImageResampling 是 PIL 图像重采样方法的枚举类型
# get_image_size 用于获取图像大小
# infer_channel_dimension_format 用于推断图像的通道维度格式
# is_scaled_image 用于判断图像是否已经缩放
# make_list_of_images 用于将单个图像转换为图像列表
# to_numpy_array 用于将图像转换为 NumPy 数组
# valid_images 用于验证图像的格式是否合法
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
# 导入通用实用函数和类型
from ...utils import TensorType, is_vision_available, logging

# 如果图像处理模块可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 PerceiverImageProcessor 类，继承自 BaseImageProcessor 类
class PerceiverImageProcessor(BaseImageProcessor):
    r"""
    构建 Perceiver 图像处理器。
    Args:
        do_center_crop (`bool`, `optional`, defaults to `True`):
            是否对图像进行中心裁剪。如果输入的尺寸在任意边上小于`crop_size`，则会使用零填充图像，然后进行中心裁剪。
            可以通过`preprocess`方法中的`do_center_crop`参数进行覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            应用中心裁剪时期望的输出尺寸。可以通过`preprocess`方法中的`crop_size`参数进行覆盖。
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像大小为`(size["height"], size["width"])`。可以通过`preprocess`方法中的`do_resize`参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            调整大小后的图像尺寸。可以通过`preprocess`方法中的`size`参数进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            定义调整大小时要使用的重采样滤波器。可以通过`preprocess`方法中的`resample`参数进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的比例因子`rescale_factor`缩放图像。可以通过`preprocess`方法中的`do_rescale`参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            定义缩放图像时要使用的比例因子。可以通过`preprocess`方法中的`rescale_factor`参数进行覆盖。
        do_normalize:
            是否对图像进行归一化。可以通过`preprocess`方法中的`do_normalize`参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果要对图像进行归一化，则使用的均值。该值为一个浮点数或与图像通道数相同长度的浮点数列表。
            可以通过`preprocess`方法中的`image_mean`参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果要对图像进行归一化，则使用的标准差。该值为一个浮点数或与图像通道数相同长度的浮点数列表。
            可以通过`preprocess`方法中的`image_std`参数进行覆盖。
    """

    # 模型输入的名称为"pixel_values"
    model_input_names = ["pixel_values"]
    # 这是一个图像预处理类的初始化方法
    def __init__(
        self,
        # 是否执行中心裁剪
        do_center_crop: bool = True,
        # 中心裁剪的尺寸大小
        crop_size: Dict[str, int] = None,
        # 是否执行调整大小
        do_resize: bool = True,
        # 调整大小的目标尺寸
        size: Dict[str, int] = None,
        # 调整大小使用的重采样方法
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 是否执行缩放
        do_rescale: bool = True,
        # 缩放因子
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否执行归一化
        do_normalize: bool = True,
        # 图像的均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像的标准差
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        # 调用父类的 __init__ 方法
        super().__init__(**kwargs)
        # 如果没有设置裁剪尺寸，使用默认值 {"height": 256, "width": 256}
        crop_size = crop_size if crop_size is not None else {"height": 256, "width": 256}
        # 确保 crop_size 是一个字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        # 如果没有设置调整大小的目标尺寸，使用默认值 {"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 确保 size 是一个字典
        size = get_size_dict(size)
    
        # 设置实例属性
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
    
    # 这是一个执行中心裁剪的方法
    def center_crop(
        self,
        # 输入图像
        image: np.ndarray,
        # 裁剪尺寸
        crop_size: Dict[str, int],
        # 可选的目标尺寸
        size: Optional[int] = None,
        # 可选的输入数据格式
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 可选的输入数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 定义函数，用于对图像进行中心裁剪
    def center_crop(image: np.ndarray, crop_size: Dict[str, int], size: Dict[str, int] = None,
                    data_format: Union[str, ChannelDimension] = None,
                    input_data_format: Union[str, ChannelDimension] = None, **kwargs) -> np.ndarray:
        """
        Center crop an image to `(size["height"] / crop_size["height"] * min_dim, size["width"] / crop_size["width"] *
        min_dim)`. Where `min_dim = min(size["height"], size["width"])`.
    
        If the input size is smaller than `crop_size` along any edge, the image will be padded with zeros and then
        center cropped.
    
        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`Dict[str, int]`):
                Desired output size after applying the center crop.
            size (`Dict[str, int]`, *optional*):
                Size of the image after resizing. If not provided, the self.size attribute will be used.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    
        # 如果未提供图像尺寸，则使用 self.size 属性
        size = self.size if size is None else size
        # 获取图像的尺寸字典
        size = get_size_dict(size)
        # 获取中心裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")
    
        # 获取图像的高度和宽度
        height, width = get_image_size(image, channel_dim=input_data_format)
        # 获取图像的最小边长
        min_dim = min(height, width)
        # 计算裁剪后的高度和宽度
        cropped_height = (size["height"] / crop_size["height"]) * min_dim
        cropped_width = (size["width"] / crop_size["width"]) * min_dim
        # 返回中心裁剪后的图像
        return center_crop(
            image,
            size=(cropped_height, cropped_width),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    
    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor.resize中复制而来，将PILImageResampling.BILINEAR修改为PILImageResampling.BICUBIC
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
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
        # 将 size 参数转换为字典格式
        size = get_size_dict(size)
        # 检查 size 字典中是否包含 "height" 和 "width" 键
        if "height" not in size or "width" not in size:
            # 若不包含，则抛出值错误异常
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 提取出输出尺寸元组
        output_size = (size["height"], size["width"])
        # 调用 resize 函数，进行图像调整大小操作，并返回结果
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
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```