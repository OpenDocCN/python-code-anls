# `.\models\deit\image_processing_deit.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发本软件
# 没有任何保证或条件，无论是明示还是默示
# 有关特定语言管理权限和局限性的详细信息，请参阅许可证
"""Image processor class for DeiT.""" 的实现
# 导入所需的模块和类
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
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
)
from ...utils import TensorType, is_vision_available, logging


# 如果视觉模块可用，则导入 PIL 模块
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# 创建 DeiTImageProcessor 类，继承自 BaseImageProcessor
class DeiTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DeiT image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的(高度, 宽度)尺寸重置为指定的 `size`。可以被 `preprocess` 中的 `do_resize` 覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            `resize` 后的图像尺寸。可以被 `preprocess` 中的 `size` 覆盖。
        resample (`PILImageResampling` filter, *optional*, defaults to `Resampling.BICUBIC`):
            如果调整图像大小，则使用的重采样滤波器。可以被 `preprocess` 中的 `resample` 覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否进行中心裁剪图像。如果输入尺寸小于 `crop_size` 的任何边缘，则图像用0填充，然后进行中心裁剪。
            可以被 `preprocess` 中的 `do_center_crop` 覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            在应用中心裁剪时的期望输出尺寸。可以被 `preprocess` 中的 `crop_size` 覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果对图像进行重新缩放，则使用的缩放因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的缩放因子 `rescale_factor` 对图像进行重新缩放。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行标准化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果对图像进行标准化，则使用的均值。这是一个浮点数或与图像通道数量相等长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果对图像进行标准化，则使用的标准差。这是一个浮点数或与图像通道数量相等长度的浮点数列表。
            可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
    """

注释：这是一个类的初始化方法，接收一系列参数来配置图像预处理的过程。每个参数都有默认值，可以通过 `preprocess` 方法的参数来覆盖默认值。
    # 定义初始化函数，接受一些参数，并设置默认值
    def __init__(self, do_resize: bool = True, size: Optional[Union[int, Tuple[int, int]]] = None, 
                 resample: PILImageResampling = PILImageResampling.BILINEAR, do_center_crop: bool = False, 
                 crop_size: Optional[Union[int, Tuple[int, int]]] = None, do_rescale: bool = False, 
                 rescale_factor: float = 1.0, do_normalize: bool = False, image_mean: Optional[Union[Tensor, 
                 Tuple[float, float, float]]] = None, image_std: Optional[Union[Tensor, Tuple[float, float, float]]] = None,
                 **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 初始化size，若size为None，则设置默认值
        size = size if size is not None else {"height": 256, "width": 256}
        # 调用辅助函数，将size转化为字典格式
        size = get_size_dict(size)
        # 初始化crop_size，若crop_size为None，则设置默认值
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 调用辅助函数，将crop_size转化为字典格式
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 初始化各个参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor.resize复制的函数，使用PILImageResampling.BILINEAR->PILImageResampling.BICUBIC
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Union[ChannelDimension, str] = None,
        input_data_format: Union[ChannelDimension, str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.
        
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
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
        size = get_size_dict(size)  # Convert the size dictionary to the required format
        if "height" not in size or "width" not in size:  # Check if the 'height' and 'width' keys are present in the size dictionary
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # Extract the height and width values from the size dictionary
        return resize(  # Call the resize function with the specified parameters
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
        resample=None,
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



        ) -> Union[np.ndarray, TensorType]:
        """
        Preprocess the input images with optional resizing, cropping, rescaling, and normalization.

        Args:
            images (Union[np.ndarray, List[np.ndarray], Image.Image]):
                Input image or a list of input images to be preprocessed.
            do_resize (`bool`, *optional*):
                Whether to resize the input images. If `True`, the `size` argument must be provided.
            size (`Dict[str, int]`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size to which the input images
                should be resized. Required if `do_resize` is `True`.
            resample (`PILImageResampling`, *optional*, defaults to `None`):
                `PILImageResampling` filter to use when resizing the images e.g. `PILImageResampling.BICUBIC`.
            do_center_crop (`bool`, *optional*):
                Whether to perform center cropping on the input images. If `True`, the `crop_size` argument must be provided.
            crop_size (`Dict[str, int]`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the center crop. Required
                if `do_center_crop` is `True`.
            do_rescale (`bool`, *optional*):
                Whether to rescale the input images. If `True`, the `rescale_factor` argument must be provided.
            rescale_factor (`float`, *optional*):
                The factor by which to rescale the input images. Required if `do_rescale` is `True`.
            do_normalize (`bool`, *optional*):
                Whether to normalize the input images. If `True`, the `image_mean` and `image_std` arguments must be
                provided.
            image_mean (`Union[float, List[float]]`, *optional*):
                The mean value to use for image normalization. If a single float value is provided, it is used for all
                channels. If a list of float values is provided, it must have the same length as the number of image
                channels. Required if `do_normalize` is `True`.
            image_std (`Union[float, List[float]]`, *optional*):
                The standard deviation value to use for image normalization. If a single float value is provided, it is
                used for all channels. If a list of float values is provided, it must have the same length as the number
                of image channels. Required if `do_normalize` is `True`.
            return_tensors (`Optional[Union[str, TensorType]]`):
                The type of tensor to return. Can be one of: `"np.ndarray"`, `"tensorflow"`, `"pytorch"`, `"mxnet"`.
            data_format (`ChannelDimension`, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output images.
            input_data_format (`Union[str, ChannelDimension]`, *optional*):
                The channel dimension format for the input images. If unset, the channel dimension format is inferred
                from the input images.

        Returns:
            Union[np.ndarray, TensorType]: The preprocessed images.
        """
        # Preprocessing logic goes here
```