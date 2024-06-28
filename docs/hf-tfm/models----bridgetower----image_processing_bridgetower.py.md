# `.\models\bridgetower\image_processing_bridgetower.py`

```
# 定义脚本的编码格式为 UTF-8
# 版权声明，指明版权归属和保留的权利

"""BridgeTower 的图像处理器类。"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np  # 导入 NumPy 库

# 导入图像处理相关的工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, center_crop, pad, resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入通用的工具函数
from ...utils import TensorType, is_vision_available, logging

# 如果视觉功能可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取 logger 对象用于记录日志信息
logger = logging.get_logger(__name__)


# 从 transformers 模块中复制的函数定义，计算可迭代值中每个索引的最大值并返回列表
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# 从 transformers 模块中复制的函数定义，为图像创建像素掩码，其中 1 表示有效像素，0 表示填充像素
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    # 获取图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个与输出大小相同的像素掩码数组，初始值为 0
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效图像区域标记为 1
    mask[:input_height, :input_width] = 1
    return mask


# 从 transformers 模块中复制的函数定义，获取批处理中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未指定数据格式，则推断第一个图像的通道格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])
    # 如果输入数据格式为首先通道维度
    if input_data_format == ChannelDimension.FIRST:
        # 获取所有图像的形状，并取得最大的高度和宽度
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    # 如果输入数据格式为最后通道维度
    elif input_data_format == ChannelDimension.LAST:
        # 获取所有图像的形状，并取得最大的高度和宽度
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        # 如果输入数据格式既不是首先也不是最后通道维度，则引发值错误异常
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    # 返回最大的高度和宽度作为元组
    return (max_height, max_width)
# 从transformers.models.vilt.image_processing_vilt.get_resize_output_image_size复制而来的函数
def get_resize_output_image_size(
    input_image: np.ndarray,
    shorter: int = 800,
    longer: int = 1333,
    size_divisor: int = 32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(input_image, input_data_format)
    
    # 定义最小和最大的尺寸
    min_size, max_size = shorter, longer

    # 计算缩放比例
    scale = min_size / min(input_height, input_width)

    # 根据图像高度与宽度的比较来调整新的高度和宽度
    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    # 如果新的高度或宽度超过最大尺寸，则再次调整缩放比例
    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    # 四舍五入并确保高度和宽度是size_divisor的倍数
    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    # 返回新的高度和宽度作为元组
    return new_height, new_width


class BridgeTowerImageProcessor(BaseImageProcessor):
    r"""
    构建一个BridgeTower图像处理器。

    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_pad: bool = True,
        **kwargs,
    ):
        # 初始化BridgeTowerImageProcessor对象的各种属性
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_pad = do_pad
        # 其他参数传递给基类构造函数
        # kwargs 可以包含任何未在参数列表中指定的其他参数
    # 初始化图像处理器对象，接受各种参数以配置其行为
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像重置
        size: Optional[Dict[str, int]] = None,  # 图像大小参数字典，短边至少为288像素
        size_divisor: int = 32,  # 图像大小的除数，用于确保尺寸可以被32整除
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像重采样方法，默认为双三次插值
        do_rescale: bool = True,  # 是否进行图像缩放
        rescale_factor: Optional[float] = None,  # 图像缩放因子
        do_normalize: bool = True,  # 是否进行图像标准化
        image_mean: Optional[List[float]] = None,  # 图像像素均值，若未指定则使用预设值
        image_std: Optional[List[float]] = None,  # 图像像素标准差，若未指定则使用预设值
        do_pad: bool = False,  # 是否进行图像填充
        do_center_crop: bool = False,  # 是否进行图像中心裁剪
        crop_size: Optional[Tuple[int, int]] = None,  # 图像裁剪尺寸
        **kwargs,  # 其他可选参数，用于灵活配置
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            # 如果传入参数中包含"pad_and_return_pixel_mask"，则获取并移除这个参数
            do_pad = kwargs.pop("pad_and_return_pixel_mask")
    
        super().__init__(**kwargs)  # 调用父类初始化方法，传递其他参数给父类
    
        size = size if size is not None else {"shortest_edge": 288}  # 如果未指定size参数，则设定短边至少为288像素
        size = get_size_dict(size, default_to_square=False)  # 调用函数获取处理后的size字典
    
        # 将初始化参数赋值给对象的属性
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
    
        # 验证处理器参数的有效性，列出所有可能的有效键
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "size_divisor",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "do_center_crop",
            "crop_size",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 32,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Resizes the shorter side of the image to `size["shortest_edge"]` while preserving the aspect ratio. If the
        longer side is larger than the max size `(int(size["shortest_edge"] * 1333 / 800))`, the longer side is then
        resized to the max size while preserving the aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Controls the size of the output image. Should be of the form `{"shortest_edge": int}`.
            size_divisor (`int`, defaults to 32):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling` filter, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        
        Returns:
            np.ndarray: Resized image.

        Raises:
            ValueError: If `size` dictionary does not contain the key `"shortest_edge"`.

        """
        # Ensure the `size` dictionary is properly formatted for resizing
        size = get_size_dict(size, default_to_square=False)
        
        # Check if the required key "shortest_edge" exists in the size dictionary
        if "shortest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        
        # Retrieve the value of the shortest edge size from the `size` dictionary
        shorter = size["shortest_edge"]
        
        # Calculate the longer side size based on the aspect ratio constraint
        longer = int(1333 / 800 * shorter)
        
        # Compute the final output size for resizing the image
        output_size = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor, input_data_format=input_data_format
        )
        
        # Perform the actual resizing operation using the specified parameters
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image in the form `{"height": h, "width": w}`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        output_size = size["shortest_edge"]
        return center_crop(
            image,
            size=(output_size, output_size),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )


    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.

        Args:
            image (`np.ndarray`):
                Input image to be padded.
            output_size (`Tuple[int, int]`):
                Desired output size of the image in format `(height, width)`.
            constant_values (`Union[float, Iterable[float]]`, *optional*):
                Value or sequence of values to pad the image with. Default is 0.
            data_format (`ChannelDimension`, *optional*):
                Format of the output image channel dimension. If not specified, defaults to `None`.
            input_data_format (`Union[str, ChannelDimension]`, *optional*):
                Format of the input image channel dimension. If not specified, defaults to `None`.

        Returns:
            np.ndarray:
                Padded image of shape `(output_size[0], output_size[1], channels)`.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        
        # Perform padding operation with constant values
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image


    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor.pad
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            image (`np.ndarray`):
                Image to pad.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取批量图像中最大高度和宽度，并返回作为填充大小
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对每张图像进行填充处理，保证它们达到批量中最大高度和宽度的大小，使用指定的常量值进行填充
        padded_images = [
            self._pad_image(
                image,
                pad_size,
                constant_values=constant_values,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in images
        ]
        # 构建返回的数据字典，包含填充后的图像数组
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 对每张图像生成相应的像素掩码，并加入数据字典中
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # 返回一个 BatchFeature 对象，其中包含填充后的数据和指定类型的张量
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 定义图像预处理方法，接受多个参数来控制不同的预处理步骤和参数
    def preprocess(
        self,
        images: ImageInput,  # 输入的图像数据，可以是单张图像或图像列表
        do_resize: Optional[bool] = None,  # 是否进行图像尺寸调整的标志
        size: Optional[Dict[str, int]] = None,  # 调整后的图像尺寸，以字典形式表示
        size_divisor: Optional[int] = None,  # 尺寸调整时的除数，用于确保尺寸是某个数的倍数
        resample: PILImageResampling = None,  # 图像调整大小时使用的重采样方法
        do_rescale: Optional[bool] = None,  # 是否进行图像尺度调整的标志
        rescale_factor: Optional[float] = None,  # 图像尺度调整的比例因子
        do_normalize: Optional[bool] = None,  # 是否进行图像标准化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像标准化时的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准化时的标准差
        do_pad: Optional[bool] = None,  # 是否进行图像填充的标志
        do_center_crop: Optional[bool] = None,  # 是否进行图像中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 图像裁剪后的尺寸，以字典形式表示
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回数据的张量类型，如numpy数组或torch张量
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 图像数据的通道顺序，FIRST表示通道在前
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的通道顺序
        **kwargs,  # 其他未明确定义的参数，以字典形式接收
```