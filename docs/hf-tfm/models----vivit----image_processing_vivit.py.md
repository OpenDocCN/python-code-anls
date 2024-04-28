# `.\transformers\models\vivit\image_processing_vivit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，可以在遵守许可证的情况下使用该文件
# 获取许可证的副本可以访问 http://www.apache.org/licenses/LICENSE-2.0
# 被授权的软件基于 "AS IS" 基础分发，没有任何类型的保证或条件，无论是明示的还是隐含的
# 请查看许可证以了解具体语言的权限和限制

# 导入类型提示模块
from typing import Dict, List, Optional, Union

# 导入 numpy 模块，并重命名为 np
import numpy as np

# 导入 is_vision_available 函数
from transformers.utils import is_vision_available
# 导入 TensorType 类型
from transformers.utils.generic import TensorType

# 导入 image_processing_utils 模块中的 BaseImageProcessor 类和 BatchFeature 类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入 image_transforms 模块中的函数和常量
from ...image_transforms import (
    get_resize_output_image_size,
    rescale,
    resize,
    to_channel_dimension_format,
)
# 导入 image_utils 模块中的函数和常量
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
)
# 导入 logging 模块中的 logger
from ...utils import logging

# 如果 vision 模块可用，导入 PIL 模块
if is_vision_available():
    import PIL

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一个函数，将视频帧转换为批处理格式
def make_batched(videos) -> List[List[ImageInput]]:
    # 如果 videos 是列表或元组，并且第一个元素也是列表或元组，并且第一个元素是有效的图像
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    # 如果 videos 是列表或元组，并且第一个元素是有效的图像
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]

    # 如果 videos 是有效的图像
    elif is_valid_image(videos):
        return [[videos]]

    # 抛出数值错误，表明无法将 videos 转换为批处理格式
    raise ValueError(f"Could not make batched video from {videos}")


# 定义一个 VivitImageProcessor 类，继承自 BaseImageProcessor 类
class VivitImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Vivit image processor.
    # 定义函数参数
    Args:
        # 是否将图像的（高度，宽度）尺寸调整为指定尺寸，默认为 True
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        # 调整后的输出图像大小，默认为 {"shortest_edge": 256}
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        # 如果调整图像大小，要使用的重采样滤波器，默认为 Resampling.BILINEAR
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        # 是否对图像进行中心裁剪，默认为 True
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        # 在应用中心裁剪后的图像大小，默认为 {"height": 224, "width": 224}
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        # 是否按指定比例对图像进行重新缩放，默认为 True
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        # 如果重新缩放图像，要使用的比例因子，默认为 1/127.5
        rescale_factor (`int` or `float`, *optional*, defaults to `1/127.5`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        # 是否在负方向和正方向同时缩放图像，默认为 True
        offset (`bool`, *optional*, defaults to `True`):
            Whether to scale the image in both negative and positive directions. Can be overriden by the `offset` in
            the `preprocess` method.
        # 是否对图像进行归一化，默认为 True
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        # 如果对图像进行归一化，要使用的平均值，默认为 IMAGENET_STANDARD_MEAN
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        # 如果对图像进行归一化，要使用的标准差，默认为 IMAGENET_STANDARD_STD
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    # 模型输入的名称
    model_input_names = ["pixel_values"]
    # 初始化函数，设置各种参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小
        size: Dict[str, int] = None,  # 图像尺寸
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方式
        do_center_crop: bool = True,  # 是否进行中心裁剪
        crop_size: Dict[str, int] = None,  # 裁剪尺寸
        do_rescale: bool = True,  # 是否进行尺度调整
        rescale_factor: Union[int, float] = 1 / 127.5,  # 尺度因子
        offset: bool = True,  # 是否偏移
        do_normalize: bool = True,  # 是否进行归一化
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 如果未指定图像尺寸，则设置默认值
        size = size if size is not None else {"shortest_edge": 256}
        # 获取最终的图像尺寸字典
        size = get_size_dict(size, default_to_square=False)
        # 如果未指定裁剪尺寸，则设置默认值
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取最终的裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置各种参数
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.offset = offset
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 如果未指定图像均值，则采用默认值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 如果未指定图像标准差，则采用默认值

    # 调整图像大小的函数
    def resize(
        self,
        image: np.ndarray,  # 输入图像数据
        size: Dict[str, int],  # 图像尺寸
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方式
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式
        **kwargs,
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
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 根据输入的大小参数获取要调整的大小
        size = get_size_dict(size, default_to_square=False)
        # 如果大小参数中包含 "shortest_edge"，则根据最短边来调整输出图像大小
        if "shortest_edge" in size:
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        # 如果大小参数中包含 "height" 和 "width"，则设置输出图像大小
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        else:
            # 如果大小参数既不包含 "height" 也不包含 "width"，则抛出异常
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        # 返回调整大小后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # Copied from transformers.models.efficientnet.image_processing_efficientnet.EfficientNetImageProcessor.rescale
    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        offset: bool = True,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Rescale an image by a scale factor.

        If `offset` is `True`, the image has its values rescaled by `scale` and then offset by 1. If `scale` is
        1/127.5, the image is rescaled between [-1, 1].
            image = image * scale - 1

        If `offset` is `False`, and `scale` is 1/255, the image is rescaled between [0, 1].
            image = image * scale

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            offset (`bool`, *optional*):
                Whether to scale the image in both negative and positive directions.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 调用rescale方法对图像进行重新缩放
        rescaled_image = rescale(
            image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

        # 如果offset为True，则对图像进行偏移
        if offset:
            rescaled_image = rescaled_image - 1

        return rescaled_image

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
        offset: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def preprocess(
        self,
        videos: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        offset: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 对单个图像进行预处理

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")
        # 如果需要调整大小，但未指定大小或重采样方法，则引发数值错误异常

        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")
        # 如果需要居中裁剪，但未指定裁剪大小，则引发数值错误异常

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")
        # 如果需要重新缩放，但未指定缩放因子，则引发数值错误异常

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")
        # 如果需要归一化，但未指定图像均值或标准差，则引发数值错误异常

        if offset and not do_rescale:
            raise ValueError("For offset, do_rescale must also be set to True.")
        # 如果开启偏移，但未开启重新缩放，则引发数值错误异常

        # All transformations expect numpy arrays.
        # 所有转换都期望 numpy 数组

        image = to_numpy_array(image)
        # 将图像转换为 numpy 数组

        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 如果图像已经缩放，并且开启重新缩放，则发出警告，建议关闭重新缩放

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # 如果输入数据格式未指定，则推断通道维度的格式

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        # 如果需要调整大小，则调用 resize 方法

        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)
        # 如果需要居中裁剪，则调用 center_crop 方法

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, offset=offset, input_data_format=input_data_format)
        # 如果需要重新缩放，则调用 rescale 方法

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # 如果需要归一化，则调用 normalize 方法

        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 将图像转换为通道维度格式

        return image
```