# `.\models\clip\image_processing_clip.py`

```py
# 导入必要的模块和类
from typing import Dict, List, Optional, Union  # 导入类型提示所需的模块

import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

# 导入图像处理相关的工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,  # 导入将图像转换为 RGB 格式的函数
    get_resize_output_image_size,  # 导入获取调整大小后图像尺寸的函数
    resize,  # 导入调整图像大小的函数
    to_channel_dimension_format,  # 导入将图像转换为指定通道格式的函数
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,  # 导入 CLIP 模型期望的图像均值
    OPENAI_CLIP_STD,  # 导入 CLIP 模型期望的图像标准差
    ChannelDimension,  # 导入通道维度类型
    ImageInput,  # 导入图像输入类型
    PILImageResampling,  # 导入 PIL 图像的重采样方式
    infer_channel_dimension_format,  # 推断图像通道维度格式的函数
    is_scaled_image,  # 判断图像是否已经缩放的函数
    make_list_of_images,  # 创建图像列表的函数
    to_numpy_array,  # 将图像转换为 NumPy 数组的函数
    valid_images,  # 验证图像有效性的函数
    validate_kwargs,  # 验证关键字参数的函数
    validate_preprocess_arguments,  # 验证预处理参数的函数
)
from ...utils import TensorType, is_vision_available, logging  # 导入相关工具和日志记录模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# 检查视觉处理模块是否可用
if is_vision_available():
    import PIL  # 如果视觉处理可用，导入 PIL 库用于图像操作


class CLIPImageProcessor(BaseImageProcessor):
    r"""
    Constructs a CLIP image processor.
    
    """
    """
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
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法，传递额外参数
        super().__init__(**kwargs)
        
        # 如果未提供 size 参数，则默认设置为 {"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 根据提供的 size 参数获取最终的尺寸字典，保证不默认为正方形
        size = get_size_dict(size, default_to_square=False)
        
        # 如果未提供 crop_size 参数，则默认设置为 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据提供的 crop_size 参数获取最终的裁剪尺寸字典，保证默认为正方形，命名参数为 "crop_size"
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 将各个参数赋值给实例变量
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
        
        # 定义一个包含有效处理器键的列表，用于后向兼容 KOSMOS-2
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
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

        # 用于 KOSMOS-2 的向后兼容性，如果 kwargs 中包含 "use_square_size"，则将 size 调整为正方形
        if "use_square_size" in kwargs:
            self.size = {"height": size["shortest_edge"], "width": size["shortest_edge"]}

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
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
                Size of the output image. Can specify 'shortest_edge' or 'height' and 'width'.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Determine if we default to resizing to a square image
        default_to_square = True
        if "shortest_edge" in size:
            # Resize based on the shortest edge specified in size dictionary
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            # Resize based on both height and width specified in size dictionary
            size = (size["height"], size["width"])
        else:
            # If neither 'shortest_edge' nor ('height' and 'width') is provided, raise an error
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        # Calculate the output size for resizing while maintaining aspect ratio
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        
        # Perform the resizing operation
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
```