# `.\transformers\models\bit\image_processing_bit.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，声明代码版权归 The HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证要求，否则禁止使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证要求，本软件是基于“按原样”提供的，不附带任何形式的担保或条件
# 有关许可证的详细信息，请参阅许可证
"""BiT 的图像处理器类。"""

# 导入必要的模块和类型提示
from typing import Dict, List, Optional, Union

# 导入 NumPy 库并简称为 np
import numpy as np

# 导入自定义模块和函数
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

# 如果视觉功能可用，则导入 PIL 模块
if is_vision_available():
    import PIL

# 定义 BiT 图像处理器类，继承自 BaseImageProcessor
class BitImageProcessor(BaseImageProcessor):
    r"""
    构造一个 BiT 图像处理器。
```  
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
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
        do_normalize:
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    # 模型的输入名称列表，只包含像素值
    model_input_names = ["pixel_values"]
    # 初始化函数，用于创建一个新的CLIP图像处理器对象
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像调整大小，默认为True
        size: Dict[str, int] = None,  # 图像调整大小的目标尺寸，默认为None
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像调整大小时的重采样方法，默认为双三次插值
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 中心裁剪的目标尺寸，默认为None
        do_rescale: bool = True,  # 是否进行图像重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像归一化的均值，默认为OPENAI_CLIP_MEAN
        image_std: Optional[Union[float, List[float]]] = None,  # 图像归一化的标准差，默认为OPENAI_CLIP_STD
        do_convert_rgb: bool = True,  # 是否将图像转换为RGB格式，默认为True
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 如果size为None，则设置默认的调整大小目标尺寸为{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 根据参数获取调整大小的尺寸字典
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size为None，则设置默认的中心裁剪目标尺寸为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据参数获取中心裁剪的尺寸字典
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 设置各个参数的属性
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

    # 从transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize中复制的resize方法
    def resize(
        self,
        image: np.ndarray,  # 输入图像的数组表示
        size: Dict[str, int],  # 调整大小的目标尺寸
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据的格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，默认为None
        **kwargs,
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
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 默认情况下将图片调整为正方形
        default_to_square = True
        # 如果 size 中包含 "shortest_edge" 键
        if "shortest_edge" in size:
            # 将 size 设置为 shortest_edge 对应的值
            size = size["shortest_edge"]
            # 不再默认调整为正方形
            default_to_square = False
        # 如果 size 中同时包含 "height" 和 "width" 键
        elif "height" in size and "width" in size:
            # 将 size 设置为对应的高度和宽度组成的元组
            size = (size["height"], size["width"])
        # 如果 size 中既不包含 "shortest_edge" 键，也不包含 "height" 和 "width" 键
        else:
            # 抛出值错误
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
        
        # 获取调整后图片的大小
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 返回调整后的图片
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