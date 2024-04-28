# `.\transformers\models\vilt\image_processing_vilt.py`

```py
# 设定脚本的编码格式为 UTF-8

# 版权声明，声明代码版权归 The HuggingFace Inc. 团队所有，保留所有权利。
# 根据 Apache 许可证 2.0 版（"许可证"）获得许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 根据许可证分发的软件是基于"按原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的管理权限，请参阅许可证。

# 导入必要的类型和函数
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np

# 导入相关的图像处理工具函数和类
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
)
from ...utils import TensorType, is_vision_available, logging

# 如果视觉模块可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取记录器
logger = logging.get_logger(__name__)


# 定义函数：获取可迭代对象中每个索引的最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# 定义函数：生成图像的像素遮罩，其中 1 表示有效像素，0 表示填充
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
    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建与输出大小相同的零数组，作为像素遮罩
    mask = np.zeros(output_size, dtype=np.int64)
    # 将图像的有效像素部分（不包括填充）置为 1
    mask[:input_height, :input_width] = 1
    return mask


# 定义函数：获取批次图像中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未指定输入数据格式，则推断其格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据输入数据格式，获取批次图像中的最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# 定义函数：获取调整大小后的图像大小
def get_resize_output_image_size(
    input_image: np.ndarray,
```  
    # 定义一个整数类型的变量shorter，并初始化为800
    shorter: int = 800,
    # 定义一个整数类型的变量longer，并初始化为1333
    longer: int = 1333,
    # 定义一个整数类型的变量size_divisor，并初始化为32
    size_divisor: int = 32,
    # 定义一个可选的联合类型变量input_data_format，可以是字符串类型或ChannelDimension类型，初始值为None
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
# 定义一个函数，用于计算图像的新高度和宽度
def compute_new_size(shorter: int, longer: int, input_image: Union[np.ndarray, str], input_data_format: Optional[str] = None, size_divisor: int = 32) -> Tuple[int, int]:
    # 调用函数获取输入图像的高度和宽度
    input_height, input_width = get_image_size(input_image, input_data_format)
    # 将最小和最大尺寸设定为给定参数
    min_size, max_size = shorter, longer

    # 计算缩放比例，确保图像适应最小尺寸
    scale = min_size / min(input_height, input_width)

    # 根据输入图像的高度和宽度调整新的高度和宽度
    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    # 如果新的高度或宽度超过了最大尺寸，则重新调整缩放比例
    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    # 对新的高度和宽度进行四舍五入，并确保它们能被 size_divisor 整除
    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    # 返回新的高度和宽度
    return new_height, new_width


# 定义一个图像处理类，用于处理 ViLT 图像
class ViltImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViLT image processor.

    """

    # 模型的输入名称为 "pixel_values"
    model_input_names = ["pixel_values"]

    # 初始化方法，用于创建 ViLT 图像处理器实例
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
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        # 如果 "pad_and_return_pixel_mask" 在参数中，使用其值来更新 do_pad
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果 size 为 None，则设定默认的 size 字典
        size = size if size is not None else {"shortest_edge": 384}
        # 根据输入的 size 创建 size 字典，确保图像保持方形
        size = get_size_dict(size, default_to_square=False)

        # 初始化 ViLT 图像处理器的属性
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

    # 从字典中构建 ViLT 图像处理器实例的方法
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `reduce_labels` is updated if image processor
        is created using from_dict and kwargs e.g. `ViltImageProcessor.from_pretrained(checkpoint,
        pad_and_return_pixel_mask=False)`
        """
        # 复制图像处理器字典，以便修改
        image_processor_dict = image_processor_dict.copy()
        # 如果 "pad_and_return_pixel_mask" 在参数中，使用其值来更新 image_processor_dict
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        # 调用父类的 from_dict 方法来创建图像处理器实例
        return super().from_dict(image_processor_dict, **kwargs)
    # 定义 resize 方法，用于调整图像大小
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
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取调整后的尺寸字典，确保不是正方形
        size = get_size_dict(size, default_to_square=False)
        # 检查是否包含了最短边的键
        if "shortest_edge" not in size:
            # 若不包含则引发 ValueError 异常
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        # 获取最短边的长度
        shorter = size["shortest_edge"]
        # 计算最长边的长度
        longer = int(1333 / 800 * shorter)
        # 获取调整后的输出尺寸
        output_size = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor, input_data_format=input_data_format
        )
        # 返回调整大小后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image 复制而来的方法
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

        # 计算需要填充的底部和右侧像素数量
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充元组，用于指定像素填充位置
        padding = ((0, pad_bottom), (0, pad_right))
        # 对图像进行填充操作
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        # 返回经过填充的图像
        return padded_image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 定义填充函数，将 images 列表中的图像进行填充
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
```py  
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
        # 获取批量图像中最大高度和宽度，并以此确定要填充的大小
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对图像进行填充，使其在批量中的每张图像的底部和右侧都填充零，直到达到批量中最大高度和宽度的大小
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
        # 构建返回结果字典
        data = {"pixel_values": padded_images}

        # 如果设置了返回像素掩码
        if return_pixel_mask:
            # 为每个图像生成像素掩码
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 将像素掩码添加到返回结果字典中
            data["pixel_mask"] = masks

        # 返回结果批次特征
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 对输入的图像进行预处理
    def preprocess(
        self,
        # 输入的图像数据
        images: ImageInput,
        # 是否进行调整大小操作
        do_resize: Optional[bool] = None,
        # 调整大小的目标尺寸
        size: Optional[Dict[str, int]] = None,
        # 调整大小的尺寸除数
        size_divisor: Optional[int] = None,
        # 重采样方法
        resample: PILImageResampling = None,
        # 是否进行重新缩放操作
        do_rescale: Optional[bool] = None,
        # 重新缩放的因子
        rescale_factor: Optional[float] = None,
        # 是否进行归一化操作
        do_normalize: Optional[bool] = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否进行填充操作
        do_pad: Optional[bool] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
```