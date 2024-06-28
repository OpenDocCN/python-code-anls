# `.\models\tvlt\image_processing_tvlt.py`

```py
# 设定文件编码为 UTF-8
# 版权声明及保留所有权利给 HuggingFace Inc. 团队，未经许可不得使用
#
# 根据 Apache 许可证 2.0 版本使用本文件；您不得在未遵守许可证的情况下使用此文件。
# 您可以从以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“现状”提供的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证了解具体语言条款和权限限制。
"""TVLT 的图像处理类。"""

from typing import Dict, List, Optional, Union

import numpy as np  # 导入 NumPy 库

# 导入 TVLT 图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,  # 导入获取调整后图像尺寸的函数
    resize,  # 导入调整图像大小的函数
    to_channel_dimension_format,  # 导入将图像转换为通道维度格式的函数
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入 ImageNet 标准均值
    IMAGENET_STANDARD_STD,  # 导入 ImageNet 标准标准差
    ChannelDimension,  # 导入通道维度枚举
    ImageInput,  # 导入图像输入类型
    PILImageResampling,  # 导入 PIL 图像重采样方法
    infer_channel_dimension_format,  # 导入推断通道维度格式的函数
    is_scaled_image,  # 导入判断是否为缩放图像的函数
    is_valid_image,  # 导入判断是否为有效图像的函数
    to_numpy_array,  # 导入将图像转换为 NumPy 数组的函数
    valid_images,  # 导入判断有效图像的函数
    validate_kwargs,  # 导入验证关键字参数的函数
    validate_preprocess_arguments,  # 导入验证预处理参数的函数
)
from ...utils import TensorType, logging  # 导入 Tensor 类型和日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def make_batched(videos) -> List[List[ImageInput]]:
    """将输入的视频或图像列表转换为批处理列表形式。

    Args:
        videos: 输入的视频或图像列表

    Returns:
        List[List[ImageInput]]: 批处理后的视频或图像列表

    Raises:
        ValueError: 如果无法从输入中生成批处理视频
    """
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        videos_dim = np.array(videos[0]).ndim
        if videos_dim == 3:
            return [videos]
        elif videos_dim == 4:
            return videos

    elif is_valid_image(videos):
        videos_dim = np.array(videos).ndim
        if videos_dim == 3:
            return [[videos]]
        elif videos_dim == 4:
            return [videos]
        elif videos_dim == 5:
            return videos

    raise ValueError(f"Could not make batched video from {videos}")


class TvltImageProcessor(BaseImageProcessor):
    r"""
    构造一个 TVLT 图像处理器。

    此处理器可用于通过将图像转换为单帧视频来为模型准备视频或图像。

    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overridden by
            `size` in the `preprocess` method.
        patch_size (`List[int]` *optional*, defaults to [16,16]):
            The patch size of image patch embedding.
        num_frames (`int` *optional*, defaults to 8):
            The maximum number of video frames.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to 1/255):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
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
    # 定义模型输入的名称列表，包含四个元素
    model_input_names = [
        "pixel_values",        # 像素数值
        "pixel_mask",          # 像素掩码
        "pixel_values_mixed",  # 混合像素数值
        "pixel_mask_mixed",    # 混合像素掩码
    ]
    # 初始化方法，用于设置图像处理器的各种参数和属性
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，包含最短边或其他指定尺寸，默认为{"shortest_edge": 224}
        patch_size: List[int] = [16, 16],  # 图像的分块大小，默认为[16, 16]
        num_frames: int = 8,  # 处理视频时的帧数，默认为8
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪后图像的尺寸，默认为{"height": 224, "width": 224}
        do_rescale: bool = True,  # 是否进行图像像素值缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像像素值缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_MEAN,  # 图像归一化均值，默认为ImageNet标准均值
        image_std: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_STD,  # 图像归一化标准差，默认为ImageNet标准标准差
        init_mask_generator=False,  # 是否初始化遮罩生成器，默认为False
        **kwargs,  # 其他可选参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果未提供size参数，则设置默认的size字典
        size = size if size is not None else {"shortest_edge": 224}
        # 根据提供的size参数获取最终的size字典，保证其含有必要的尺寸信息
        size = get_size_dict(size, default_to_square=False)
        # 如果未提供crop_size参数，则设置默认的crop_size字典
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据提供的crop_size参数获取最终的crop_size字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 将初始化方法中的各个参数设置为对象的属性
        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        # 定义一个包含所有有效处理器键的列表，用于后续验证和使用
        self._valid_processor_keys = [
            "videos",
            "do_resize",
            "size",
            "patch_size",
            "num_frames",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "is_mixed",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 图像大小调整方法，用于调整输入图像的尺寸
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数组
        size: Dict[str, int],  # 目标图像尺寸的字典
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式参数
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式参数
        **kwargs,  # 其他可选参数
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
        # 根据 size 获取实际的大小字典，确保不是默认的正方形输出
        size = get_size_dict(size, default_to_square=False)

        # 如果 size 字典中包含 "shortest_edge" 键
        if "shortest_edge" in size:
            # 根据最短边长度调整输出图像大小
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        # 如果 size 字典中包含 "height" 和 "width" 键
        elif "height" in size and "width" in size:
            # 设置输出大小为指定的高度和宽度
            output_size = (size["height"], size["width"])
        else:
            # 如果 size 字典既不包含 "shortest_edge" 也不同时包含 "height" 和 "width" 键，抛出数值错误
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")

        # 调用 resize 函数，返回调整大小后的图像
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
    ) -> np.ndarray:
        """Preprocesses a single image."""

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)  # Convert input image to numpy array format

        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)  # Infer input data format if not provided

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)  # Resize image if required

        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)  # Perform center cropping if specified

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)  # Rescale image if specified

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)  # Normalize image if specified

        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)  # Convert image to desired channel dimension format
        return image

    def preprocess(
        self,
        videos: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        patch_size: List[int] = None,
        num_frames: int = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        is_mixed: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```