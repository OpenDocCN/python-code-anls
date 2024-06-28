# `.\models\efficientformer\image_processing_efficientformer.py`

```
# 导入所需模块和类，包括类型提示和必要的功能函数
from typing import Dict, List, Optional, Union

import numpy as np  # 导入numpy库，用于数值计算

# 导入图像处理所需的工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,  # 导入获取调整后图像尺寸的函数
    resize,  # 导入图像调整大小的函数
    to_channel_dimension_format,  # 导入将图像转换为通道维度格式的函数
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,  # 导入图像处理中的默认均值
    IMAGENET_DEFAULT_STD,   # 导入图像处理中的默认标准差
    ChannelDimension,       # 导入通道维度相关的枚举
    ImageInput,             # 导入图像输入的相关类型
    PILImageResampling,     # 导入PIL图像重采样相关的枚举
    infer_channel_dimension_format,  # 导入推断图像通道维度格式的函数
    is_batched,             # 导入判断图像是否批处理的函数
    is_scaled_image,        # 导入判断图像是否已经缩放的函数
    to_numpy_array,         # 导入将图像转换为numpy数组的函数
    valid_images,           # 导入验证图像是否有效的函数
    validate_kwargs,        # 导入验证关键字参数的函数
    validate_preprocess_arguments,  # 导入验证预处理参数的函数
)
from ...utils import TensorType, logging  # 导入TensorType和日志记录相关的工具函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象


class EfficientFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a EfficientFormer image processor.
    
    """

    def __init__(self):
        super().__init__()  # 调用父类BaseImageProcessor的构造函数，初始化基础图像处理器
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    # 初始化方法，设置预处理参数
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        crop_size: Dict[str, int] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    # 初始化函数，继承父类并设置参数
    def __init__(
        self,
        do_resize: bool = False,
        do_rescale: bool = False,
        do_normalize: bool = False,
        do_center_crop: bool = False,
        size: Optional[Dict[str, int]] = None,
        crop_size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        # 调用父类初始化函数，并传递额外的参数
        super().__init__(**kwargs)
        # 如果未指定 size 参数，默认设置为 224x224
        size = size if size is not None else {"height": 224, "width": 224}
        # 使用函数处理 size 参数，确保格式正确
        size = get_size_dict(size)
        # 如果未指定 crop_size 参数，默认设置为 224x224，并采用默认的正方形裁剪
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 使用函数处理 crop_size 参数，采用默认的正方形裁剪
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
    
        # 设置各种图像处理标志
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_center_crop = do_center_crop
        # 设置裁剪尺寸
        self.crop_size = crop_size
        # 设置图像大小
        self.size = size
        # 设置重采样方式
        self.resample = resample
        # 设置图像缩放因子
        self.rescale_factor = rescale_factor
        # 设置图像均值，默认为 IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 设置图像标准差，默认为 IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 设置有效的处理键列表
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
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The resized image.
        """
        # 获取经过处理后的尺寸字典
        size = get_size_dict(size)

        # 如果 size 字典中有 "shortest_edge" 键，根据最短边调整图片大小
        if "shortest_edge" in size:
            # 调用函数计算输出图片的大小
            size = get_resize_output_image_size(
                image, size=size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
            # 如果 size 中有 "height" 和 "width" 键，将它们作为尺寸
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果 size 中既不含 "shortest_edge" 也不含完整的尺寸信息，则抛出异常
            raise ValueError(f"Size must contain 'height' and 'width' keys or 'shortest_edge' key. Got {size.keys()}")
        # 调用 resize 函数进行图片大小调整，并返回调整后的图片
        return resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
```