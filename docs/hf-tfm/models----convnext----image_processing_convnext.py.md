# `.\models\convnext\image_processing_convnext.py`

```py
# 导入必要的模块和函数
from typing import Dict, List, Optional, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库，用于处理数组数据

# 导入图像处理相关的工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    center_crop,  # 导入中心裁剪函数
    get_resize_output_image_size,  # 导入获取调整后图像尺寸函数
    resize,  # 导入调整图像尺寸函数
    to_channel_dimension_format,  # 导入转换为通道维度格式函数
)
# 导入图像处理工具函数和常量
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入图像标准均值常量
    IMAGENET_STANDARD_STD,  # 导入图像标准标准差常量
    ChannelDimension,  # 导入通道维度枚举
    ImageInput,  # 导入图像输入类
    PILImageResampling,  # 导入 PIL 图像重采样枚举
    infer_channel_dimension_format,  # 导入推断通道维度格式函数
    is_scaled_image,  # 导入判断是否为缩放图像函数
    make_list_of_images,  # 导入创建图像列表函数
    to_numpy_array,  # 导入转换为 NumPy 数组函数
    valid_images,  # 导入验证图像函数
    validate_kwargs,  # 导入验证关键字参数函数
    validate_preprocess_arguments,  # 导入验证预处理参数函数
)
# 导入通用工具函数和类型检查相关函数
from ...utils import TensorType, is_vision_available, logging  # 导入张量类型和可视化库是否可用函数


if is_vision_available():  # 检查是否可用视觉处理库
    import PIL  # 导入 PIL 库，用于图像处理


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象


class ConvNextImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ConvNeXT image processor.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 384}`):
            Resolution of the output image after `resize` is applied. If `size["shortest_edge"]` >= 384, the image is
            resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the image will
            be matched to `int(size["shortest_edge"]/crop_pct)`, after which the image is cropped to
            `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`. Can
            be overriden by `size` in the `preprocess` method.
        crop_pct (`float` *optional*, defaults to 224 / 256):
            Percentage of the image to crop. Only has an effect if `do_resize` is `True` and size < 384. Can be
            overriden by `crop_pct` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overriden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overriden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
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

    # 列出模型输入名称列表，包含像素值
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        crop_pct: float = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 如果参数中未提供 size，则设置一个默认的 size 字典，确保 size 中至少包含键 "shortest_edge" 且值为 384
        size = size if size is not None else {"shortest_edge": 384}
        # 根据传入的 size 参数，获取标准化的尺寸字典，确保返回一个方形的尺寸字典
        size = get_size_dict(size, default_to_square=False)

        # 初始化对象属性
        self.do_resize = do_resize  # 是否进行图像调整大小的标志
        self.size = size  # 图像调整大小后的目标尺寸字典
        # 如果 crop_pct 为 None，则设置一个默认的裁剪比例，默认为 224/256
        self.crop_pct = crop_pct if crop_pct is not None else 224 / 256
        self.resample = resample  # 图像调整大小时使用的重采样方法，默认为双线性插值
        self.do_rescale = do_rescale  # 是否进行图像重新缩放的标志
        self.rescale_factor = rescale_factor  # 图像重新缩放的因子，默认为 1/255
        self.do_normalize = do_normalize  # 是否进行图像标准化的标志
        # 如果 image_mean 为 None，则使用预定义的 IMAGENET_STANDARD_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 如果 image_std 为 None，则使用预定义的 IMAGENET_STANDARD_STD
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        # 设置一个有效的处理器关键字列表，用于后续验证和处理
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "crop_pct",
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
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"shortest_edge": int}`, specifying the size of the output image. If
                `size["shortest_edge"]` >= 384 image is resized to `(size["shortest_edge"], size["shortest_edge"])`.
                Otherwise, the smaller edge of the image will be matched to `int(size["shortest_edge"] / crop_pct)`,
                after which the image is cropped to `(size["shortest_edge"], size["shortest_edge"])`.
            crop_pct (`float`):
                Percentage of the image to crop. Only has an effect if size < 384.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # Ensure size dictionary includes the 'shortest_edge' key
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" not in size:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        
        shortest_edge = size["shortest_edge"]

        if shortest_edge < 384:
            # Calculate the resized shortest edge based on crop percentage
            resize_shortest_edge = int(shortest_edge / crop_pct)
            # Determine the output size after resizing with maintaining aspect ratio
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=input_data_format
            )
            # Resize the image using specified parameters
            image = resize(
                image=image,
                size=resize_size,
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
            # Crop the resized image to (shortest_edge, shortest_edge)
            return center_crop(
                image=image,
                size=(shortest_edge, shortest_edge),
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
        else:
            # Resize the image without cropping when size is 384 or larger
            return resize(
                image,
                size=(shortest_edge, shortest_edge),
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
    # 图像预处理函数，接受多种参数用于处理图像
    def preprocess(
        self,
        # 输入的图像数据，可以是单个图像或图像列表
        images: ImageInput,
        # 是否执行调整图像大小的操作，默认为None
        do_resize: bool = None,
        # 调整图像大小的目标尺寸，字典格式，包括宽度和高度
        size: Dict[str, int] = None,
        # 裁剪图像的百分比，用于裁剪中心区域，默认为None
        crop_pct: float = None,
        # 重采样方法，例如缩放图像时使用的插值方法，默认为None
        resample: PILImageResampling = None,
        # 是否执行图像缩放操作，默认为None
        do_rescale: bool = None,
        # 缩放因子，用于调整图像大小，默认为None
        rescale_factor: float = None,
        # 是否执行图像标准化操作，默认为None
        do_normalize: bool = None,
        # 图像的均值，用于标准化操作，可以是单个值或列表形式，默认为None
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像的标准差，用于标准化操作，可以是单个值或列表形式，默认为None
        image_std: Optional[Union[float, List[float]]] = None,
        # 返回的数据类型，可以是字符串或张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据的通道格式，首通道或者其他，默认为首通道
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的通道格式，可以是字符串或通道维度，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数，用于扩展预处理功能
        **kwargs,
```