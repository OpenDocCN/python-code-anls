# `.\models\videomae\image_processing_videomae.py`

```py
# 指定编码为UTF-8，确保源文件可以正确解析中文和其他非ASCII字符
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# 根据Apache License, Version 2.0许可协议，这段代码版权归HuggingFace Inc.团队所有
# 除非遵循许可协议的规定，否则不得使用此文件
# 可以从以下链接获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于"按现状"提供，不提供任何明示或暗示的保证或条件
# 请查阅许可协议以获取更多详细信息
"""Image processor class for VideoMAE."""


from typing import Dict, List, Optional, Union

import numpy as np

# 从image_processing_utils模块导入BaseImageProcessor、BatchFeature和get_size_dict函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 从image_transforms模块导入get_resize_output_image_size、resize和to_channel_dimension_format函数
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
# 从image_utils模块导入IMAGENET_STANDARD_MEAN、IMAGENET_STANDARD_STD、ChannelDimension、
# ImageInput、PILImageResampling、infer_channel_dimension_format、is_scaled_image、is_valid_image、
# to_numpy_array、valid_images、validate_kwargs和validate_preprocess_arguments函数
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
    validate_kwargs,
    validate_preprocess_arguments,
)
# 从utils模块导入TensorType、is_vision_available和logging函数
from ...utils import TensorType, is_vision_available, logging

# 如果视觉处理可用，导入PIL库
if is_vision_available():
    import PIL

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义make_batched函数，用于将视频处理成批处理的图像序列
def make_batched(videos) -> List[List[ImageInput]]:
    # 检查videos是否是列表或元组，且第一个元素也是列表或元组，且第一个图像是有效的
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    # 检查videos是否是列表或元组，且第一个元素是有效的图像
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]

    # 检查videos是否是有效的图像
    elif is_valid_image(videos):
        return [[videos]]

    # 如果无法构建批处理视频，则抛出值错误异常
    raise ValueError(f"Could not make batched video from {videos}")


# 定义VideoMAEImageProcessor类，继承自BaseImageProcessor类
class VideoMAEImageProcessor(BaseImageProcessor):
    r"""
    Constructs a VideoMAE image processor.
    # 定义函数参数说明文档，描述了预处理图像的可选参数及其默认值和用途
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）尺寸至指定 `size`。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            调整后图像的尺寸。图像的最短边将被调整至 `size["shortest_edge"]`，同时保持原始图像的长宽比。
            可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            调整图像尺寸时使用的重采样滤波器。可以被 `preprocess` 方法中的 `resample` 参数覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪至指定 `crop_size`。可以被 `preprocess` 方法中的 `do_center_crop` 参数覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            应用中心裁剪后的图像尺寸。可以被 `preprocess` 方法中的 `crop_size` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的缩放因子 `rescale_factor` 进行图像缩放。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果进行图像缩放，定义要使用的缩放因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            归一化时使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            归一化时使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """
    
    # 定义模型输入名称列表，只包含一个元素 "pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化函数，用于设置图像处理的各种参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行大小调整，默认为True
        size: Dict[str, int] = None,  # 图像尺寸的字典，默认为{"shortest_edge": 224}
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪尺寸的字典，默认为{"height": 224, "width": 224}
        do_rescale: bool = True,  # 是否进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为IMAGENET_STANDARD_MEAN
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为IMAGENET_STANDARD_STD
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类初始化方法
        super().__init__(**kwargs)
        
        # 如果没有传入size，则使用默认的{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 调用函数get_size_dict处理size，确保参数合法性
        size = get_size_dict(size, default_to_square=False)
        
        # 如果没有传入crop_size，则使用默认的{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 调用函数get_size_dict处理crop_size，确保参数合法性
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 初始化对象的各个属性
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        
        # 验证处理器关键字列表，用于后续数据处理
        self._valid_processor_keys = [
            "videos",
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
        # 根据 size 参数获取确切的图像尺寸，如果设置了 default_to_square=False，则不强制成正方形
        size = get_size_dict(size, default_to_square=False)
        
        # 如果 size 字典中包含 "shortest_edge" 键，根据最短边的长度调整图像大小
        if "shortest_edge" in size:
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        # 如果 size 字典同时包含 "height" 和 "width" 键，直接使用指定的高度和宽度
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        else:
            # 如果 size 字典既不包含 "shortest_edge" 也不包含 "height" 和 "width"，抛出异常
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        
        # 调用 resize 函数，对图像进行调整大小操作，传入指定的参数
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
        """
        Preprocesses a single image.
        """
        # Validate preprocessing arguments based on provided options
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

        # Convert image to numpy array format for consistent handling
        image = to_numpy_array(image)

        # Warn if attempting to rescale already scaled images unnecessarily
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # Infer input data format if not explicitly provided
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # Resize image if specified
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        # Perform center cropping if specified
        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        # Rescale image if specified
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # Normalize image pixel values if specified
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # Convert image to the desired channel dimension format
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

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
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Ultimate method to preprocess images or videos with flexible options.
        """
```