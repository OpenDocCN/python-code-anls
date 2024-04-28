# `.\models\efficientformer\image_processing_efficientformer.py`

```py
# 设定文件编码格式为 utf-8
# 版权声明
# 授权许可
# 获取许可副本
# 如果法律要求，或者以书面形式约定，授权许可的软件以“原样”分发，没有任何明示或暗示的担保或条件
# 限制授权许可的特定语言，授权许可对权限和限制具体的语言进行了说明

# 导入必要的模块和类
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling
from ...image_utils import infer_channel_dimension_format, is_batched, is_scaled_image, to_numpy_array, valid_images
from ...utils import TensorType, logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 EfficientFormerImageProcessor 类，继承自 BaseImageProcessor
class EfficientFormerImageProcessor(BaseImageProcessor):
    r"""
    构建一个 EfficientFormer 图像处理器
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        crop_size: Dict[str, int] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    # 设置初始化方法，参数为 None
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置图像大小，默认为 224x224
        size = size if size is not None else {"height": 224, "width": 224}
        # 转换图像大小字典为标准格式
        size = get_size_dict(size)
        # 设置裁剪大小，默认为 224x224
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 转换裁剪大小字典为标准格式，使其保持正方形
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 设置是否进行调整大小的标志
        self.do_resize = do_resize
        # 设置是否进行重新缩放的标志
        self.do_rescale = do_rescale
        # 设置是否进行标准化的标志
        self.do_normalize = do_normalize
        # 设置是否进行中心裁剪的标志
        self.do_center_crop = do_center_crop
        # 设置裁剪大小
        self.crop_size = crop_size
        # 设置图像大小
        self.size = size
        # 设置重采样方法，默认为双线性插值
        self.resample = resample
        # 设置重新缩放因子
        self.rescale_factor = rescale_factor
        # 设置图像均值，默认为 ImageNet 默认均值
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 设置图像标准差，默认为 ImageNet 默认标准差
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    # 设置图像调整大小方法
    def resize(
        # 输入图像数组
        self,
        image: np.ndarray,
        # 调整后的图像大小
        size: Dict[str, int],
        # 重采样方法，默认为双线性插值
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        # 数据格式，默认为 None
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The resized image.
        """
        # 获取指定大小的字典形式的图片尺寸
        size = get_size_dict(size)

        if "shortest_edge" in size:
            # 根据最短边来调整图像大小
            size = get_resize_output_image_size(
                image, size=size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
            # size = get_resize_output_image_size(image, size["shortest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            # 如果包含了指定的高度和宽度，则直接使用这两个值作为大小
            size = (size["height"], size["width"])
        else:
            # 如果size不包含'height'和'width'键或者'shortest_edge'键，则报错
            raise ValueError(f"Size must contain 'height' and 'width' keys or 'shortest_edge' key. Got {size.keys()}")
        # 返回调整后的图像
        return resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```