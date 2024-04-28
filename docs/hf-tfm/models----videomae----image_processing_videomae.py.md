# `.\transformers\models\videomae\image_processing_videomae.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，保留所有权利
#
# 根据 Apache 许可证，需要遵守许可证规定使用本文件，许可证详情请访问 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件为“按原样”分发，不提供任何明示或暗示的保证或条件
# 请参阅许可证以获取许可证下特定语言规定的权限和限制
"""VideoMAE""" 的图像处理器类

from typing import Dict, List, Optional, Union  # 导入类型提示相关库

import numpy as np  # 导入NumPy库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理工具相关函数
from ...image_transforms import (  # 导入图像变换相关函数
    get_resize_output_image_size,  # 获取调整大小后的图像尺寸
    resize,  # 调整图像大小
    to_channel_dimension_format,  # 将图像转换为指定通道维度格式
)
from ...image_utils import (  # 导入图像工具相关函数
    IMAGENET_STANDARD_MEAN,  # 导入标准 ImageNet 图像均值
    IMAGENET_STANDARD_STD,  # 导入标准 ImageNet 图像标准差
    ChannelDimension,  # 通道维度
    ImageInput,  # 图像输入类型
    PILImageResampling,  # PIL 图像重采样方式
    infer_channel_dimension_format,  # 推断通道维度格式
    is_scaled_image,  # 检查是否为缩放图像
    is_valid_image,  # 检查是否为有效图像
    to_numpy_array,  # 将图像转换为 NumPy 数组
    valid_images,  # 有效图像列表
)
from ...utils import TensorType, is_vision_available, logging  # 导入工具函数

if is_vision_available():  # 如果有图像处理库可用
    import PIL  # 导入PIL库

logger = logging.get_logger(__name__)  # 获取日志记录器


def make_batched(videos) -> List[List[ImageInput]]:  # 创建批次图像
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):  # 如果输入为列表或元组，且第一个元素也是列表或元组，且第一个元素是有效图像
        return videos  # 返回原始列表

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):  # 如果输入为列表或元组，且第一个元素是有效图像
        return [videos]  # 返回包含原始列表的新列表

    elif is_valid_image(videos):  # 如果输入是有效图像
        return [[videos]]  # 返回包含新列表的新列表

    raise ValueError(f"Could not make batched video from {videos}")  # 抛出数值错误异常，无法生成批量视频


class VideoMAEImageProcessor(BaseImageProcessor):  # VideoMAE 图像处理器类
    r"""
    构建一个 VideoMAE 图像处理器
    # 定义函数参数及其默认值，用于图像预处理
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            # 是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            # 调整大小后的输出图像尺寸。图像的最短边将调整为 `size["shortest_edge"]`，同时保持原始图像的宽高比。
            # 可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            # 调整图像大小时使用的重采样滤波器。可以被 `preprocess` 方法中的 `resample` 参数覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            # 是否对图像进行中心裁剪以符合指定的 `crop_size`。可以被 `preprocess` 方法中的 `do_center_crop` 参数覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            # 在应用中心裁剪后的图像大小。可以被 `preprocess` 方法中的 `crop_size` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            # 是否按指定的比例 `rescale_factor` 对图像进行重新缩放。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            # 如果重新缩放图像，则定义要使用的比例因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            # 是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            # 如果对图像进行归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            # 如果对图像进行归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """

    # 模型输入名称列表
    model_input_names = ["pixel_values"]
    # 初始化方法，用于设置图像预处理参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志，默认为 True
        size: Dict[str, int] = None,  # 图像调整大小的目标尺寸字典，默认为 None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志，默认为 True
        crop_size: Dict[str, int] = None,  # 中心裁剪的目标尺寸字典，默认为 None
        do_rescale: bool = True,  # 是否进行图像重新缩放的标志，默认为 True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为 1/255
        do_normalize: bool = True,  # 是否进行图像标准化的标志，默认为 True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像标准化均值，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准化标准差，默认为 None
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果 size 参数为 None，则设置默认的图像调整尺寸字典
        size = size if size is not None else {"shortest_edge": 224}
        # 将 size 参数转换为标准的图像尺寸字典
        size = get_size_dict(size, default_to_square=False)
        # 如果 crop_size 参数为 None，则设置默认的中心裁剪尺寸字典
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 将 crop_size 参数转换为标准的中心裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置各种参数值
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 如果 image_mean 为 None，则使用默认值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 如果 image_std 为 None，则使用默认值

    # 图像调整方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,  # 待调整的图像数组
        size: Dict[str, int],  # 目标尺寸字典
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
        **kwargs,  # 其他关键字参数
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"shortest_edge": s}`, the output image will have its
                shortest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 根据指定的大小参数获取要输出的图像大小字典
        size = get_size_dict(size, default_to_square=False)
        # 如果最短边在大小字典中，则根据最短边的大小进行调整
        if "shortest_edge" in size:
            # 根据指定的最短边大小计算输出图像的大小
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        # 如果同时指定了高度和宽度，则直接使用这两个参数
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        # 如果以上条件都不满足，则抛出错误
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        # 调用resize函数进行图像大小调整，并返回调整后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 该函数用于预处理单个图像
    def preprocess_image(
        image,
        do_resize: bool = False,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = False,
        rescale_factor: float = None,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> np.ndarray:
        # 如果需要缩放但没有指定大小和采样方法，则引发错误
        if do_resize and (size is None or resample is None):
            raise ValueError("Size and resample must be specified if do_resize is True.")
        
        # 如果需要中心裁剪但没有指定裁剪大小，则引发错误
        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")
        
        # 如果需要缩放但没有指定缩放因子，则引发错误
        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")
        
        # 如果需要归一化但没有指定均值和标准差，则引发错误
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")
        
        # 将输入转换为 NumPy 数组
        image = to_numpy_array(image)
        
        # 如果图像已经被缩放，并且需要再次缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        
        # 如果未指定输入数据格式，则推断通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        
        # 执行缩放操作
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        
        # 执行中心裁剪操作
        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)
        
        # 执行缩放操作
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
        
        # 执行归一化操作
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        
        # 将图像转换为所需的通道维度格式
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        
        return image
```