# `.\transformers\models\tvp\image_processing_tvp.py`

```py
# 定义编码格式为 UTF-8
# 版权声明
#
# 根据 Apache 许可证 2.0 版（"许可证"）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 在没有任何明示或暗示的情况下，
# 根据许可证分发的软件都是按“原样”提供的，
# 没有任何形式的保证或条件，
# 查看许可证以了解特定语言下的权限和限制。
"""TVP 的图像处理类。"""

# 导入类型提示工具
from typing import Dict, Iterable, List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np

# 导入图像处理工具
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像转换函数
from ...image_transforms import (
    PaddingMode,
    flip_channel_order,
    pad,
    resize,
    to_channel_dimension_format,
)
# 导入图像工具
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    is_valid_image,
    to_numpy_array,
    valid_images,
)
# 导入视觉工具和日志
from ...utils import TensorType, is_vision_available, logging

# 如果视觉工具可用
if is_vision_available():
    # 导入 Python Imaging Library
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)


# 从 transformers.models.vivit.image_processing_vivit.make_batched 复制函数
def make_batched(videos) -> List[List[ImageInput]]:
    # 如果 videos 是列表或元组，且其元素也是列表或元组，并且第一个元素是有效图像
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        # 直接返回 videos
        return videos

    # 如果 videos 是列表或元组，并且其第一个元素是有效图像
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        # 将 videos 包装为列表
        return [videos]

    # 如果 videos 是有效图像
    elif is_valid_image(videos):
        # 将 videos 包装为列表，并再次包装为列表
        return [[videos]]

    # 如果无法创建批处理视频
    raise ValueError(f"Could not make batched video from {videos}")


# 获取调整大小后输出图像的大小
def get_resize_output_image_size(
    input_image: np.ndarray,
    max_size: int = 448,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    # 获取输入图像的高度和宽度
    height, width = get_image_size(input_image, input_data_format)
    # 如果高度大于等于宽度
    if height >= width:
        # 计算高度与宽度之比
        ratio = width * 1.0 / height
        # 新高度为最大尺寸
        new_height = max_size
        # 新宽度按比例计算
        new_width = new_height * ratio
    # 如果宽度大于高度
    else:
        # 计算宽度与高度之比
        ratio = height * 1.0 / width
        # 新宽度为最大尺寸
        new_width = max_size
        # 新高度按比例计算
        new_height = new_width * ratio
    # 返回新的图像尺寸
    size = (int(new_height), int(new_width))
    return size


# TVP 图像处理器类，继承自基础图像处理器类
class TvpImageProcessor(BaseImageProcessor):
    r"""
    构建一个 TVP 图像处理器。

    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化函数，用于创建图像预处理器对象
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像尺寸调整的标志，默认为True
        size: Dict[str, int] = None,  # 调整后图像的尺寸，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整时使用的重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪后图像的尺寸，默认为None
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像像素值缩放的因子，默认为1/255
        do_pad: bool = True,  # 是否进行图像填充的标志，默认为True
        pad_size: Dict[str, int] = None,  # 填充后图像的尺寸，默认为None
        constant_values: Union[float, Iterable[float]] = 0,  # 图像填充时使用的常数值或常数值列表，默认为0
        pad_mode: PaddingMode = PaddingMode.CONSTANT,  # 图像填充时使用的填充模式，默认为CONSTANT
        do_normalize: bool = True,  # 是否进行图像归一化的标志，默认为True
        do_flip_channel_order: bool = True,  # 是否进行通道顺序翻转的标志，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像归一化时使用的均值，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像归一化时使用的标准差，默认为None
        **kwargs,  # 其他参数，以字典形式接收
    ) -> None:  # 函数返回None
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果size为None，则设置默认的size参数为{"longest_edge": 448}
        size = size if size is not None else {"longest_edge": 448}
        # 如果crop_size为None，则设置默认的crop_size参数为{"height": 448, "width": 448}
        crop_size = crop_size if crop_size is not None else {"height": 448, "width": 448}
        # 如果pad_size为None，则设置默认的pad_size参数为{"height": 448, "width": 448}
        pad_size = pad_size if pad_size is not None else {"height": 448, "width": 448}

        # 初始化各种参数
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.constant_values = constant_values
        self.pad_mode = pad_mode
        self.do_normalize = do_normalize
        self.do_flip_channel_order = do_flip_channel_order
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 如果image_mean为None，则设置为IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 如果image_std为None，则设置为IMAGENET_STANDARD_STD

    # 图像调整函数，用于调整图像的尺寸和重采样
    def resize(
        self,
        image: np.ndarray,  # 输入图像的数组表示
        size: Dict[str, int],  # 调整后图像的尺寸
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整时使用的重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，默认为None
        **kwargs,  # 其他参数，以字典形式接收
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"longest_edge": s}`, the output image will have its
                longest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Get the size dictionary with default_to_square set to False
        size = get_size_dict(size, default_to_square=False)
        
        # Check if size has both 'height' and 'width' keys
        if "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        # If size has 'longest_edge' key, calculate output size based on it
        elif "longest_edge" in size:
            output_size = get_resize_output_image_size(image, size["longest_edge"], input_data_format)
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'longest_edge' as keys. Got {size.keys()}")
        
        # Resize the image using the defined parameters and return the resized image
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # Padding function to add borders to an image to reach the desired size
    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        pad_mode: PaddingMode = PaddingMode.CONSTANT,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 定义一个对图像进行填充的函数，将图像填充到给定的大小
    def pad_image(
        image: np.ndarray,
        pad_size: Dict[str, int],
        constant_values: Union[float, Iterable[float]],
        pad_mode: PaddingMode = PaddingMode.CONSTANT,
        data_format: ChannelDimension or str = None,
        input_data_format: ChannelDimension or str = None,
    ):
        # 获取图像的高度和宽度
        height, width = get_image_size(image, channel_dim=input_data_format)
        # 获取填充后的最大高度和宽度
        max_height = pad_size.get("height", height)
        max_width = pad_size.get("width", width)

        # 计算需要填充的右侧和底部像素数
        pad_right, pad_bottom = max_width - width, max_height - height
        # 如果需要填充的像素数小于0，则抛出异常
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("The padding size must be greater than image size")

        # 构建填充的方式
        padding = ((0, pad_bottom), (0, pad_right))
        # 对图像进行填充操作
        padded_image = pad(
            image,
            padding,
            mode=pad_mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return padded_image

    # 预处理图像的函数
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
        do_pad: bool = True,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = None,
        pad_mode: PaddingMode = None,
        do_normalize: bool = None,
        do_flip_channel_order: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 如果需要调整大小，要求size和resample参数必须被指定
        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")
        
        # 如果需要进行中心裁剪，要求crop_size参数必须被指定
        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")
        
        # 如果需要进行重新缩放，要求rescale_factor参数必须被指定
        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")
        
        # 如果需要进行填充，要求pad_size参数必须被指定
        if do_pad and pad_size is None:
            raise ValueError("Padding size must be specified if do_pad is True.")
        
        # 如果需要进行归一化，要求image_mean和image_std参数必须被指定
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")
        
        # 所有的转换都需要将图像转换为numpy数组
        image = to_numpy_array(image)
        
        # 如果需要调整大小，使用指定的参数进行调整大小操作
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        
        # 如果需要进行中心裁剪，使用指定的参数进行中心裁剪操作
        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)
        
        # 如果需要重新缩放，使用指定的参数进行重新缩放操作
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
        
        # 如果需要归一化，使用指定的参数进行归一化操作
        if do_normalize:
            image = self.normalize(
                image=image.astype(np.float32), mean=image_mean, std=image_std, input_data_format=input_data_format
            )
        
        # 如果需要进行填充，使用指定的参数进行填充操作
        if do_pad:
            image = self.pad_image(
                image=image,
                pad_size=pad_size,
                constant_values=constant_values,
                pad_mode=pad_mode,
                input_data_format=input_data_format,
            )
        
        # 预训练模型要求图像为BGR格式，而不是RGB格式
        if do_flip_channel_order:
            image = flip_channel_order(image=image, input_data_format=input_data_format)
        
        # 将图��的通道维度转换为指定的数据格式和输入通道维度
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        
        # 返回处理后的图像
        return image
    # 图像预处理函数，用于对输入的视频进行预处理
    def preprocess(
        self,
        # 视频数据，可以是单个图像，图像列表或图像列表的列表
        videos: Union[ImageInput, List[ImageInput], List[List[ImageInput]]],
        # 是否进行调整大小操作
        do_resize: bool = None,
        # 调整大小的目标尺寸
        size: Dict[str, int] = None,
        # 重采样方法
        resample: PILImageResampling = None,
        # 是否进行中心裁剪操作
        do_center_crop: bool = None,
        # 中心裁剪的目标尺寸
        crop_size: Dict[str, int] = None,
        # 是否进行尺度调整操作
        do_rescale: bool = None,
        # 尺度调整的因子
        rescale_factor: float = None,
        # 是否进行填充操作
        do_pad: bool = None,
        # 填充的目标尺寸
        pad_size: Dict[str, int] = None,
        # 填充的常数值
        constant_values: Union[float, Iterable[float]] = None,
        # 填充的方式
        pad_mode: PaddingMode = None,
        # 是否进行归一化操作
        do_normalize: bool = None,
        # 是否翻转通道顺序
        do_flip_channel_order: bool = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 返回张量的格式
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
```