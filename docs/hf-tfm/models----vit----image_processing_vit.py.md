# `.\models\vit\image_processing_vit.py`

```
# 导入必要的库和模块，包括类型提示、NumPy等
from typing import Dict, List, Optional, Union

import numpy as np

# 导入自定义的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数和常量
from ...image_transforms import resize, to_channel_dimension_format
# 导入图像处理相关的工具函数和常量，如均值、标准差、通道格式等
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入通用的工具函数，如日志记录
from ...utils import TensorType, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个 ViT 图像处理器类，继承自 BaseImageProcessor 类
class ViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
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
    # 设置模型输入名称为"pixel_values"
    model_input_names = ["pixel_values"]

    # 初始化函数，设置各个参数的默认值，可以通过`preprocess`方法中的对应参数进行覆盖
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法，并传递所有关键字参数
        super().__init__(**kwargs)
        # 如果 size 参数不为 None，则使用指定的尺寸；否则使用默认尺寸 {"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 根据 size 获取一个标准化的尺寸字典
        size = get_size_dict(size)
        # 初始化是否执行 resize 操作的标志
        self.do_resize = do_resize
        # 初始化是否执行 rescale 操作的标志
        self.do_rescale = do_rescale
        # 初始化是否执行 normalize 操作的标志
        self.do_normalize = do_normalize
        # 将 size 存储到对象属性中
        self.size = size
        # 设定图像 resize 时使用的 resample 方法
        self.resample = resample
        # 设定图像 rescale 时的缩放因子
        self.rescale_factor = rescale_factor
        # 如果 image_mean 不为 None，则使用给定的 image_mean；否则使用 IMAGENET_STANDARD_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 如果 image_std 不为 None，则使用给定的 image_std；否则使用 IMAGENET_STANDARD_STD
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        # 初始化一个有效的处理器键列表，用于检查处理器的有效性
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
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
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)  # 获取调整大小后的字典格式大小
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # 获取调整后的输出大小
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
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        ):
        """
        Preprocesses a batch of images including resizing, rescaling, normalization, and tensor conversion.

        Args:
            images (`ImageInput`):
                Batch of input images.
            do_resize (`bool`, *optional*):
                Whether to resize the images. If `True`, resizing will be performed according to `size`.
            size (`Dict[str, int]`, *optional*):
                Dictionary specifying the target size for resizing each image in the batch.
            resample (`PILImageResampling`, *optional*):
                Resampling method to use for resizing, default is `PILImageResampling.BILINEAR`.
            do_rescale (`bool`, *optional*):
                Whether to rescale the images. If `True`, images will be scaled by `rescale_factor`.
            rescale_factor (`float`, *optional*):
                Scaling factor for rescaling images.
            do_normalize (`bool`, *optional*):
                Whether to normalize the images.
            image_mean (`float` or `List[float]`, *optional*):
                Mean values for normalization.
            image_std (`float` or `List[float]`, *optional*):
                Standard deviation values for normalization.
            return_tensors (`str` or `TensorType`, *optional*):
                If specified, converts the output to the desired tensor type (e.g., `"torch"` or `"tensorflow"`).
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output images.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input images.

        Returns:
            `np.ndarray` or `TensorType`: Preprocessed batch of images.
        """
```