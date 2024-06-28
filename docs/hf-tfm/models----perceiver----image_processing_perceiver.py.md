# `.\models\perceiver\image_processing_perceiver.py`

```py
# 导入必要的模块和函数
from typing import Dict, List, Optional, Union  # 导入类型提示相关的模块

import numpy as np  # 导入NumPy库，用于数值计算

# 导入与图像处理相关的工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  
# 从自定义模块中导入基础图像处理器、批量特征和获取大小字典函数

from ...image_transforms import center_crop, resize, to_channel_dimension_format  
# 从自定义模块中导入中心裁剪、调整大小和转换通道维度格式的函数

from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,  # 导入ImageNet图像默认均值
    IMAGENET_DEFAULT_STD,   # 导入ImageNet图像默认标准差
    ChannelDimension,       # 导入通道维度枚举
    ImageInput,             # 导入图像输入类
    PILImageResampling,     # 导入PIL图像重采样枚举
    get_image_size,         # 导入获取图像尺寸的函数
    infer_channel_dimension_format,  # 导入推断通道维度格式的函数
    is_scaled_image,        # 导入判断是否为缩放图像的函数
    make_list_of_images,    # 导入生成图像列表的函数
    to_numpy_array,         # 导入转换为NumPy数组的函数
    valid_images,           # 导入验证图像函数
    validate_kwargs,        # 导入验证关键字参数的函数
    validate_preprocess_arguments,  # 导入验证预处理参数的函数
)

from ...utils import TensorType, is_vision_available, logging  # 导入张量类型、判断视觉库是否可用和日志记录相关的模块和函数

if is_vision_available():
    import PIL  # 如果视觉库可用，则导入PIL库

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象
    # 定义函数参数和默认值，用于控制图像预处理的各个步骤和参数
    Args:
        do_center_crop (`bool`, `optional`, defaults to `True`):
            是否进行中心裁剪图像。如果输入尺寸小于 `crop_size` 的任何边，图像将被填充为零，然后进行中心裁剪。
            可以被 `preprocess` 方法中的 `do_center_crop` 参数覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            应用中心裁剪时的期望输出尺寸。可以被 `preprocess` 方法中的 `crop_size` 参数覆盖。
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像大小为 `(size["height"], size["width"])`。
            可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            调整大小后的图像尺寸。可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            定义在调整图像大小时使用的重采样滤波器。
            可以被 `preprocess` 方法中的 `resample` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的比例因子 `rescale_factor` 进行重新缩放图像。
            可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，则定义要使用的比例因子。
            可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize:
            是否对图像进行归一化。
            可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果归一化图像，则使用的平均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果归一化图像，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """
    
    # 定义模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化函数，设置图像预处理参数和默认值
    def __init__(
        self,
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪尺寸字典，可以为空
        do_resize: bool = True,  # 是否进行调整大小，默认为True
        size: Dict[str, int] = None,  # 调整大小的目标尺寸字典，可以为空
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 调整大小的插值方法，默认为双三次插值
        do_rescale: bool = True,  # 是否进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可以为空
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，可以为空
        **kwargs,  # 其他未指定参数
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 如果未指定裁剪尺寸，则使用默认的裁剪尺寸
        crop_size = crop_size if crop_size is not None else {"height": 256, "width": 256}
        # 根据指定的裁剪尺寸参数名称获取尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        # 如果未指定调整大小的目标尺寸，则使用默认的目标尺寸
        size = size if size is not None else {"height": 224, "width": 224}
        # 获取调整大小的尺寸字典
        size = get_size_dict(size)

        # 初始化对象的属性
        self.do_center_crop = do_center_crop  # 是否进行中心裁剪
        self.crop_size = crop_size  # 裁剪尺寸字典
        self.do_resize = do_resize  # 是否进行调整大小
        self.size = size  # 调整大小的目标尺寸字典
        self.resample = resample  # 调整大小的插值方法
        self.do_rescale = do_rescale  # 是否进行重新缩放
        self.rescale_factor = rescale_factor  # 重新缩放的因子
        self.do_normalize = do_normalize  # 是否进行归一化
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN  # 图像均值
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD  # 图像标准差
        # 有效的处理器关键字列表
        self._valid_processor_keys = [
            "images", "do_center_crop", "crop_size", "do_resize", "size", 
            "resample", "do_rescale", "rescale_factor", "do_normalize", 
            "image_mean", "image_std", "return_tensors", "data_format", 
            "input_data_format"
        ]
    ) -> np.ndarray:
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
        # 如果没有提供特定的尺寸参数，则使用默认的 self.size 属性
        size = self.size if size is None else size
        # 根据给定的 size 获取一个规范化的尺寸字典
        size = get_size_dict(size)
        # 根据给定的 crop_size 获取一个规范化的尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 获取输入图片的高度和宽度，并根据输入数据格式确定通道维度
        height, width = get_image_size(image, channel_dim=input_data_format)
        # 计算输入图片中较小的维度作为 min_dim
        min_dim = min(height, width)
        # 计算裁剪后的高度和宽度，确保按比例缩放
        cropped_height = (size["height"] / crop_size["height"]) * min_dim
        cropped_width = (size["width"] / crop_size["width"]) * min_dim
        # 调用 center_crop 函数进行中心裁剪，并返回裁剪后的图片
        return center_crop(
            image,
            size=(cropped_height, cropped_width),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制，修改了 resample 参数的默认值为 PILImageResampling.BICUBIC
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
        # Ensure `size` dictionary contains required keys
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        
        # Prepare output size tuple
        output_size = (size["height"], size["width"])
        
        # Call the `resize` function to resize the image
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