# `.\models\deit\image_processing_deit.py`

```py
# 引入必要的依赖库和模块，包括类型提示和图像处理工具
from typing import Dict, List, Optional, Union  # 导入类型提示
import numpy as np  # 导入 NumPy 库

# 导入图像处理相关的工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入图像标准均值
    IMAGENET_STANDARD_STD,   # 导入图像标准方差
    ChannelDimension,        # 导入通道维度
    ImageInput,              # 导入图像输入类型
    PILImageResampling,      # 导入 PIL 图像重采样方式
    infer_channel_dimension_format,  # 推断通道维度格式
    is_scaled_image,         # 判断是否为缩放图像
    make_list_of_images,     # 创建图像列表
    to_numpy_array,          # 转换为 NumPy 数组
    valid_images,            # 验证图像有效性
    validate_kwargs,         # 验证关键字参数
    validate_preprocess_arguments,  # 验证预处理参数
)
from ...utils import TensorType, is_vision_available, logging  # 导入其他必要工具

# 如果视觉工具可用，导入 PIL 库
if is_vision_available():
    import PIL

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 DeiTImageProcessor 类，继承自 BaseImageProcessor 类
class DeiTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DeiT image processor.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the image after `resize`. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling` filter, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in `preprocess`.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired output size when applying center-cropping. Can be overridden by `crop_size` in `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
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
    # 定义模型输入名称列表，只包含一个元素"pixel_values"
    model_input_names = ["pixel_values"]

    # 初始化方法，设置各种预处理参数的默认值
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PIL.Image.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
    # 定义构造函数，继承自父类，接受关键字参数
    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = False,
        rescale_factor: float = 1.0,
        do_normalize: bool = False,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        # 调用父类的构造函数，传入所有关键字参数
        super().__init__(**kwargs)
        
        # 如果 size 参数为 None，则设定默认值 {"height": 256, "width": 256}
        size = size if size is not None else {"height": 256, "width": 256}
        # 调用 get_size_dict 函数，规范化 size 字典的格式
        size = get_size_dict(size)
        
        # 如果 crop_size 参数为 None，则设定默认值 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 调用 get_size_dict 函数，规范化 crop_size 字典的格式，并指定参数名 "crop_size"
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设定是否进行 resize 操作的标志
        self.do_resize = do_resize
        # 设定图像尺寸的参数
        self.size = size
        # 设定图像 resize 时的插值方法
        self.resample = resample
        # 设定是否进行中心裁剪操作的标志
        self.do_center_crop = do_center_crop
        # 设定裁剪尺寸的参数
        self.crop_size = crop_size
        # 设定是否进行图像 rescale 的标志
        self.do_rescale = do_rescale
        # 设定图像 rescale 的比例因子
        self.rescale_factor = rescale_factor
        # 设定是否进行图像 normalize 的标志
        self.do_normalize = do_normalize
        # 如果未指定图像均值，则使用预设的 IMAGENET_STANDARD_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 如果未指定图像标准差，则使用预设的 IMAGENET_STANDARD_STD
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        # 设定有效的处理器关键字列表，用于验证
        self._valid_processor_keys = [
            "images",
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
    
    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制的函数，设定插值方法为 PILImageResampling.BICUBIC
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
        size = get_size_dict(size)  # 调用函数获取处理后的尺寸信息
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # 获取输出图像的尺寸
        return resize(
            image,
            size=output_size,  # 调整图像尺寸为指定的大小
            resample=resample,  # 使用指定的重采样方法
            data_format=data_format,  # 设置输出图像的通道维度格式
            input_data_format=input_data_format,  # 设置输入图像的通道维度格式
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
```