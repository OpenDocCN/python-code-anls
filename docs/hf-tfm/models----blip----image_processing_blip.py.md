# `.\transformers\models\blip\image_processing_blip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制
"""Image processor class for BLIP.""" 的注释

# 导入必要的模块和类型提示
from typing import Dict, List, Optional, Union
# 导入 numpy 库
import numpy as np

# 导入自定义的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像转换函数
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
# 导入图像处理工具函数
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
# 导入通用工具函数和类型
from ...utils import TensorType, is_vision_available, logging

# 如果视觉模块可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 BLIP 图像处理器类，继承自 BaseImageProcessor 类
class BlipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BLIP image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    # Define the names of the model input
    model_input_names = ["pixel_values"]

    # Initialize the ImagePreprocessing class with specified parameters
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        # 调用父类的构造方法初始化对象
        super().__init__(**kwargs)
        # 如果传入的尺寸参数为None，则设定默认尺寸为384x384，并确保为方形
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=True)

        # 设置是否进行图片大小调整的标志
        self.do_resize = do_resize
        # 设置图片的大小
        self.size = size
        # 设置图片的重采样方法
        self.resample = resample
        # 设置是否进行图片大小缩放的标志
        self.do_rescale = do_rescale
        # 设置图片缩放的因子
        self.rescale_factor = rescale_factor
        # 设置是否进行图片数据归一化的标志
        self.do_normalize = do_normalize
        # 设置图片的均值，如果未提供则使用默认值
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        # 设置图片的标准差，如果未提供则使用默认值
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        # 设置是否进行RGB转换的标志
        self.do_convert_rgb = do_convert_rgb

    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor.resize方法复制，修改了重采样方法为BICUBIC
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
        将图像调整大小为`(size["height"], size["width"])`。

        Args:
            image (`np.ndarray`):
                要调整大小的图像。
            size (`Dict[str, int]`):
                以`{"height": int, "width": int}`格式指定输出图像的大小的字典。
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                调整图像大小时要使用的`PILImageResampling`滤镜，例如`PILImageResampling.BICUBIC`。
            data_format (`ChannelDimension` or `str`, *optional*):
                输出图像的通道维度格式。如果未设置，则使用输入图像的通道维度格式。可以是以下之一：
                - `"channels_first"`或`ChannelDimension.FIRST`：图像以(num_channels, height, width)格式表示。
                - `"channels_last"`或`ChannelDimension.LAST`：图像以(height, width, num_channels)格式表示。
                - `"none"`或`ChannelDimension.NONE`：图像以(height, width)格式表示。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。如果未设置，则从输入图像中推断出通道维度格式。可以是以下之一：
                - `"channels_first"`或`ChannelDimension.FIRST`：图像以(num_channels, height, width)格式表示。
                - `"channels_last"`或`ChannelDimension.LAST`：图像以(height, width, num_channels)格式表示。
                - `"none"`或`ChannelDimension.NONE`：图像以(height, width)格式表示。

        Returns:
            `np.ndarray`: 调整大小后的图像。
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
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
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```