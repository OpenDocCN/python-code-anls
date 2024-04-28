# `.\transformers\models\chinese_clip\image_processing_chinese_clip.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 OFA-Sys 团队作者和 HuggingFace 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""Chinese-CLIP 的图像处理器类。"""

# 导入必要的模块和类型提示
from typing import Dict, List, Optional, Union

import numpy as np

# 导入自定义的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
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
from ...utils import TensorType, is_vision_available, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果视觉模块可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 定义 ChineseCLIPImageProcessor 类，继承自 BaseImageProcessor
class ChineseCLIPImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Chinese-CLIP image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
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
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    # 定义模型输入的名称列表，只包含像素值
    model_input_names = ["pixel_values"]
    # 初始化方法，设置各种参数
    def __init__(
        self,
        # 是否进行调整大小
        do_resize: bool = True,
        # 图像大小
        size: Dict[str, int] = None,
        # 重采样方法
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 是否进行中心裁剪
        do_center_crop: bool = True,
        # 裁剪大小
        crop_size: Dict[str, int] = None,
        # 是否进行重新缩放
        do_rescale: bool = True,
        # 重新缩放因子
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否进行归一化
        do_normalize: bool = True,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否转换为 RGB
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果没有指定图像大小，则设置默认值
        size = size if size is not None else {"shortest_edge": 224}
        # 获取图像大小字典
        size = get_size_dict(size, default_to_square=False)
        # 如果没有指定裁剪大小，则设置默认值
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取裁剪大小字典
        crop_size = get_size_dict(crop_size)

        # 设置各种参数
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

    # 调整图像大小的方法
    def resize(
        self,
        # 输入图像
        image: np.ndarray,
        # 目标大小
        size: Dict[str, int],
        # 重采样方法
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 数据格式
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # 根据输入的尺寸字典获取尺寸信息，不进行强制方形化
        size = get_size_dict(size, default_to_square=False)
        # 根据输入图像和尺寸信息计算输出图像的尺寸
        output_size = get_resize_output_image_size(
            image, size=(size["height"], size["width"]), default_to_square=False, input_data_format=input_data_format
        )
        # 调整图像大小
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
```