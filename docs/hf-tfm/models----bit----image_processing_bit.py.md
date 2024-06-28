# `.\models\bit\image_processing_bit.py`

```py
# 导入所需的模块和库
from typing import Dict, List, Optional, Union  # 导入类型提示相关的模块

import numpy as np  # 导入NumPy库，用于处理数组和矩阵操作

# 导入图像处理相关的实用工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,                     # 导入转换为RGB格式的函数
    get_resize_output_image_size,       # 导入获取调整后图像大小的函数
    resize,                             # 导入图像调整大小的函数
    to_channel_dimension_format,        # 导入将图像转换为通道维度格式的函数
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,                   # 导入OpenAI CLIP平均值
    OPENAI_CLIP_STD,                    # 导入OpenAI CLIP标准差
    ChannelDimension,                   # 导入通道维度枚举
    ImageInput,                         # 导入图像输入类型
    PILImageResampling,                 # 导入PIL图像重采样方式
    infer_channel_dimension_format,     # 推断通道维度格式的函数
    is_scaled_image,                    # 判断是否为缩放图像的函数
    make_list_of_images,                # 将图像转换为图像列表的函数
    to_numpy_array,                     # 将图像转换为NumPy数组的函数
    valid_images,                       # 检查图像是否有效的函数
    validate_kwargs,                    # 验证关键字参数的函数
    validate_preprocess_arguments,      # 验证预处理参数的函数
)
from ...utils import TensorType, is_vision_available, logging  # 导入Tensor类型，视觉库是否可用标志和日志记录功能

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 如果视觉库可用，则导入PIL模块
if is_vision_available():
    import PIL  # 导入PIL库，用于图像处理
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize:
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    # 模型输入的名称，这里指定了一个名称为 "pixel_values" 的输入
    model_input_names = ["pixel_values"]
    # 初始化函数，设置各种预处理参数的默认值
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小
        size: Dict[str, int] = None,  # 图像大小字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法
        do_center_crop: bool = True,  # 是否进行中心裁剪
        crop_size: Dict[str, int] = None,  # 裁剪大小字典
        do_rescale: bool = True,  # 是否进行重新缩放
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子
        do_normalize: bool = True,  # 是否进行标准化
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差
        do_convert_rgb: bool = True,  # 是否进行 RGB 转换
        **kwargs,
    ) -> None:
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 如果未指定图像大小，则设置默认值
        size = size if size is not None else {"shortest_edge": 224}
        # 获取图像大小字典
        size = get_size_dict(size, default_to_square=False)
        # 如果未指定裁剪大小，则设置默认值
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取裁剪大小字典
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 设置各参数的值
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从 transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize 方法复制而来
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数组
        size: Dict[str, int],  # 图像大小的字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 默认使用正方形调整大小
        default_to_square = True
        # 如果 `size` 字典中包含 'shortest_edge' 键
        if "shortest_edge" in size:
            # 将 `size` 设为 `size["shortest_edge"]`，即最短边的长度
            size = size["shortest_edge"]
            # 关闭默认正方形调整大小选项
            default_to_square = False
        # 如果 `size` 字典中包含 'height' 和 'width' 键
        elif "height" in size and "width" in size:
            # 将 `size` 设为包含高度和宽度的元组
            size = (size["height"], size["width"])
        else:
            # 如果 `size` 字典中既没有 'shortest_edge' 也没有同时包含 'height' 和 'width' 键，则抛出异常
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
        
        # 获取调整大小后的输出图像尺寸
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 调用 resize 函数进行图像调整大小操作，并返回调整大小后的图像
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
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
```