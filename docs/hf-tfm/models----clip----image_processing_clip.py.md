# `.\transformers\models\clip\image_processing_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 此代码版权归 2022 年的 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本（"许可证"）进行许可；
# 除非符合许可证，否则不得使用此文件
# 您可以获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取有关特定语言的许可证


"""CLIP 的图像处理类。"""
# 导入必要的库和模块
from typing import Dict, List, Optional, Union

import numpy as np

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

# 检查视觉库是否可用
if is_vision_available():
    # 如果可用，导入必要的 PIL 库
    import PIL
    # 定义函数参数及默认值
    Args:
        # 是否调整图像大小至指定尺寸，默认为 True
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        # 调整后图像的尺寸，默认为 {"shortest_edge": 224}
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        # 调整图像大小时使用的重采样滤波器，默认为 Resampling.BICUBIC
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        # 是否对图像进行中心裁剪，默认为 True
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        # 中心裁剪后输出图像的尺寸，默认为 224
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        # 是否对图像进行重新缩放，默认为 True
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        # 重新缩放时使用的比例因子，默认为 1/255
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        # 是否对图像进行标准化，默认为 True
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        # 标准化时使用的均值，默认为 [0.48145466, 0.4578275, 0.40821073]
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        # 标准化时使用的标准差，默认为 [0.26862954, 0.26130258, 0.27577711]
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        # 是否将图像转换为 RGB 格式，默认为 True
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    # 模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化方法，设置各种参数的默认值
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小操作，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，默认为None
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为BICUBIC
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪大小的字典，默认为None
        do_rescale: bool = True,  # 是否进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为None
        image_std: Optional[Union[float, List[float]] = None,  # 图像标准差，默认为None
        do_convert_rgb: bool = True,  # 是否转换为RGB格式，默认为True
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果size为None，则设置默认值为{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 获取调整大小后的字典
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size为None，则设置默认值为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取裁剪大小的字典
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 设置各个参数的数值
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

        # 为了向后兼容KOSMOS-2
        if "use_square_size" in kwargs:
            # 如果关键字参数中包含"use_square_size"，则设置size为正方形
            self.size = {"height": size["shortest_edge"], "width": size["shortest_edge"]}

    # 调整图像大小的方法
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数组
        size: Dict[str, int],  # 调整后的大小字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为BICUBIC
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,  # 其他关键字参数
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
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 默认将图片调整为正方形
        default_to_square = True
        # 如果 size 中包含 "shortest_edge" 键
        if "shortest_edge" in size:
            # 将 size 设为 "shortest_edge" 对应的值
            size = size["shortest_edge"]
            # 取消默认调整为正方形的设置
            default_to_square = False
        # 如果 size 中同时包含 "height" 和 "width" 键
        elif "height" in size and "width" in size:
            # 将 size 设为由 "height" 和 "width" 对应的值组成的元组
            size = (size["height"], size["width"])
        else:
            # 抛出异常，要求 size 中必须包含要求的键
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        # 获取调整后的输出尺寸
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 返回调整尺寸后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 图像预处理函数
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