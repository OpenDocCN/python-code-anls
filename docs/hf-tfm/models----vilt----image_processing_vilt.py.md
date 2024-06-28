# `.\models\vilt\image_processing_vilt.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有
# 在 Apache 许可证 2.0 版本下许可使用本文件，详情请见 http://www.apache.org/licenses/LICENSE-2.0
# 如果不符合许可证要求，则不得使用本文件
# 该脚本从 Vision 能力可用性角度导入所需模块和函数
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# 导入图像处理相关工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入日志工具
from ...utils import TensorType, is_vision_available, logging

# 如果 Vision 能力可用，则导入 PIL 模块
if is_vision_available():
    import PIL

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值中各索引位置的最大值列表。
    """
    return [max(values_i) for values_i in zip(*values)]


def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    创建图像的像素掩码，其中 1 表示有效像素，0 表示填充像素。

    Args:
        image (`np.ndarray`):
            要创建像素掩码的图像。
        output_size (`Tuple[int, int]`):
            掩码的输出大小。
    """
    # 获取图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个与输出大小相同的零矩阵
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效像素的区域设为 1
    mask[:input_height, :input_width] = 1
    return mask


def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批量图像中所有图像的最大高度和宽度。
    """
    # 如果未指定输入数据格式，则推断第一个图像的通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据通道维度格式计算最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)
# 定义函数以计算调整后图像的大小，确保长宽比例并且符合给定的尺寸要求
def get_resize_output_image_size(
    input_image: np.ndarray,
    shorter: int = 800,  # 最短边调整后的目标长度，默认为800像素
    longer: int = 1333,   # 最长边调整后的目标长度，默认为1333像素
    size_divisor: int = 32,  # 调整后的图像大小应为32的倍数
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:  # 返回调整后的图像高度和宽度

    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(input_image, input_data_format)

    # 确定调整后的最小和最大尺寸
    min_size, max_size = shorter, longer

    # 计算缩放比例，以确保最小边调整到指定长度
    scale = min_size / min(input_height, input_width)

    # 根据图像的长宽比进行调整
    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    # 如果调整后的最大边超过了指定的最大尺寸，则再次缩放图像大小
    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    # 将浮点数的像素大小四舍五入为整数
    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)

    # 将调整后的图像大小调整为指定的大小倍数
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    # 返回调整后的图像高度和宽度
    return new_height, new_width


# ViltImageProcessor 类，继承自 BaseImageProcessor 类
class ViltImageProcessor(BaseImageProcessor):
    r"""
    构建 ViLT 图像处理器。

    """

    # 模型的输入名称列表
    model_input_names = ["pixel_values"]

    # 初始化方法
    def __init__(
        self,
        do_resize: bool = True,  # 是否调整图像大小，默认为 True
        size: Dict[str, int] = None,  # 图像尺寸字典，默认为 {"shortest_edge": 384}
        size_divisor: int = 32,  # 图像大小的调整倍数，默认为32
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # PIL 图像重采样方法，默认为双三次插值
        do_rescale: bool = True,  # 是否对图像进行重新缩放，默认为 True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为 1/255
        do_normalize: bool = True,  # 是否对图像进行标准化，默认为 True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为 IMAGENET_STANDARD_MEAN
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为 IMAGENET_STANDARD_STD
        do_pad: bool = True,  # 是否对图像进行填充，默认为 True
        **kwargs,  # 其他关键字参数
    ) -> None:  # 返回空值

        # 如果关键字参数中包含 "pad_and_return_pixel_mask"，则将 do_pad 设置为该值并从 kwargs 中移除该项
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 调用父类的初始化方法，传入所有的关键字参数
        super().__init__(**kwargs)

        # 如果未提供 size 参数，则设置默认的 size 字典，以非正方形图像为默认
        size = size if size is not None else {"shortest_edge": 384}
        size = get_size_dict(size, default_to_square=False)

        # 初始化对象的各种属性
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad

        # 定义有效的处理器关键字列表，用于后续验证和处理
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
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    # 重写基类的 `from_dict` 方法，以确保在使用 `from_dict` 创建图像处理器时更新 `reduce_labels`
    # 如果通过 `ViltImageProcessor.from_pretrained(checkpoint, pad_and_return_pixel_mask=False)` 创建图像处理器
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        # 复制输入的字典，以免修改原始参数
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中包含 `pad_and_return_pixel_mask` 参数，则更新到 `image_processor_dict` 中，并从 kwargs 中移除
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        # 调用基类的 `from_dict` 方法，传入更新后的参数 `image_processor_dict` 和额外的 kwargs
        return super().from_dict(image_processor_dict, **kwargs)

    # 调整图像大小的方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Resizes the shorter side of the image to `size["shortest_edge"]` while preserving the aspect ratio. If the
        longer side is larger than the max size `(int(`size["shortest_edge"]` * 1333 / 800))`, the longer side is then
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
        """
        # 根据输入的 size 字典获取调整后的大小，确保不改变图像的长宽比
        size = get_size_dict(size, default_to_square=False)
        # 检查 `size` 字典是否包含 `shortest_edge` 键，如果不包含则抛出 ValueError
        if "shortest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        # 获取调整后的最短边和根据比例计算的最长边
        shorter = size["shortest_edge"]
        longer = int(1333 / 800 * shorter)
        # 计算最终的输出大小
        output_size = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor, input_data_format=input_data_format
        )
        # 调用实际的图像调整函数 `resize`，传入图像、输出大小、重采样方法和格式参数，以及额外的 kwargs
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    # 使用 self 参数作为方法的第一个参数，表示该方法是类的一部分
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
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        
        # 获取输出图像的高度和宽度
        output_height, output_width = output_size
        
        # 计算需要在图像底部和右侧填充的像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        
        # 构建填充元组，((top_pad, bottom_pad), (left_pad, right_pad))
        padding = ((0, pad_bottom), (0, pad_right))
        
        # 调用 pad 方法进行图像填充，返回填充后的图像
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        
        # 返回填充后的图像
        return padded_image

    # 使用 self 参数作为方法的第一个参数，表示该方法是类的一部分
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
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
        # Determine the maximum height and width in the batch of images
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # Pad each image in the batch to match `pad_size`
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
        
        # Prepare the data dictionary to hold padded images
        data = {"pixel_values": padded_images}

        # Optionally, compute and add pixel masks to the data dictionary
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return BatchFeature object containing padded images and optional masks
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 对输入的图像数据进行预处理的方法
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,  # 是否进行调整大小的标志
        size: Optional[Dict[str, int]] = None,  # 调整大小的目标尺寸
        size_divisor: Optional[int] = None,  # 调整大小的除数
        resample: PILImageResampling = None,  # 重采样方法
        do_rescale: Optional[bool] = None,  # 是否进行重新缩放的标志
        rescale_factor: Optional[float] = None,  # 重新缩放的因子
        do_normalize: Optional[bool] = None,  # 是否进行归一化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像归一化的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像归一化的标准差
        do_pad: Optional[bool] = None,  # 是否进行填充的标志
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量的格式
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 数据格式，通道维度优先
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式
        **kwargs,  # 其它可选参数
```