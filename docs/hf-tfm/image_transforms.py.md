# `.\transformers\image_transforms.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入警告模块
import warnings
# 导入类型提示相关模块
from typing import Iterable, List, Optional, Tuple, Union
# 导入 numpy 模块
import numpy as np

# 导入图像处理相关模块
from .image_utils import (
    ChannelDimension,
    ImageInput,
    get_channel_dimension_axis,
    get_image_size,
    infer_channel_dimension_format,
)
# 导入工具函数
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
# 导入模块检查函数
from .utils.import_utils import (
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    requires_backends,
)

# 如果图像处理模块可用
if is_vision_available():
    # 导入 PIL 模块
    import PIL
    # 导入图像处理相关模块
    from .image_utils import PILImageResampling

# 如果 PyTorch 可用
if is_torch_available():
    # 导入 PyTorch 模块
    import torch

# 如果 TensorFlow 可用
if is_tf_available():
    # 导入 TensorFlow 模块
    import tensorflow as tf

# 如果 Flax 可用
if is_flax_available():
    # 导入 JAX 模块
    import jax.numpy as jnp

# 将图像转换为指定的通道维度格式
def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    # 如果输入图像不是 numpy 数组，则抛出异常
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    # 如果未提供输入通道维度，则从输入图像中推断
    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    # 将目标通道维度设置为指定的通道维度
    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    # 根据目标通道维度对图像进行转置
    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image

# 重新缩放图像
def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    # 检查输入的图像是否为 np.ndarray 类型，如果不是则抛出数值错误
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    # 将图像按照给定的比例进行缩放
    rescaled_image = image * scale
    # 如果指定了数据格式，则将缩放后的图像转换为指定的数据格式
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    # 将缩放后的图像转换为指定的数据类型
    rescaled_image = rescaled_image.astype(dtype)

    # 返回缩放后的图像
    return rescaled_image
def _rescale_for_pil_conversion(image):
    """
    检测图像在转换为 PIL 图像之前是否需要重新缩放。

    假设如果图像的类型是 `np.float`，并且所有值都在 0 到 1 之间，则需要重新缩放。
    """
    # 如果图像的数据类型是 uint8，则不需要重新缩放
    if image.dtype == np.uint8:
        do_rescale = False
    # 如果图像的所有值都是整数且在 0 到 255 之间，则不需要重新缩放
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    # 如果图像的所有值都在 0 到 1 之间，则需要重新缩放
    elif np.all(0 <= image) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "torch.Tensor", "tf.Tensor", "jnp.ndarray"],
    do_rescale: Optional[bool] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> "PIL.Image.Image":
    """
    将 `image` 转换为 PIL 图像。可选择重新缩放并在需要时将通道维度放回到最后一个轴。

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor` or `tf.Tensor`):
            要转换为 `PIL.Image` 格式的图像。
        do_rescale (`bool`, *optional*):
            是否应用缩放因子（使像素值为 0 到 255 之间的整数）。如果图像类型是浮点类型，并且转换为 `int` 会导致精度损失，则默认为 `True`，否则为 `False`。
        input_data_format (`ChannelDimension`, *optional*):
            输入图像的通道维度格式。如果未设置，则将使用从输入推断出的格式。

    Returns:
        `PIL.Image.Image`: 转换后的图像。
    """
    requires_backends(to_pil_image, ["vision"])

    # 如果输入是 PIL 图像，则直接返回
    if isinstance(image, PIL.Image.Image):
        return image

    # 在转换为 PIL 图像之前，将所有张量转换为 numpy 数组
    if is_torch_tensor(image) or is_tf_tensor(image):
        image = image.numpy()
    elif is_jax_tensor(image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # 如果通道已移动到第一个维度，则将其放回到最后
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # 如果只有一个通道，则挤压它，否则 PIL 无法处理
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image
    # 如果需要，根据 PIL.Image 的要求将图像重新缩放到 0 到 255 之间的范围
    # 如果 do_rescale 为 None，则调用 _rescale_for_pil_conversion 函数进行判断是否需要重新缩放
    do_rescale = _rescale_for_pil_conversion(image) if do_rescale is None else do_rescale

    # 如果需要重新缩放图像
    if do_rescale:
        # 使用 rescale 函数将图像缩放到 0 到 255 的范围
        image = rescale(image, 255)

    # 将图像转换为 uint8 类型的数组
    image = image.astype(np.uint8)
    # 使用 PIL.Image.fromarray 函数将数组转换为 PIL 图像对象，并返回
    return PIL.Image.fromarray(image)
# 从torchvision的调整大小逻辑中借鉴：https://github.com/pytorch/vision/blob/511924c1ced4ce0461197e5caa64ce5b9e558aab/torchvision/transforms/functional.py#L366
def get_resize_output_image_size(
    # 输入图像的 numpy 数组
    input_image: np.ndarray,
    # 目标大小，可以是整数、元组、列表或者单个整数
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    # 默认是否转换为方形图像
    default_to_square: bool = True,
    # 图像的最大尺寸
    max_size: Optional[int] = None,
    # 输入图像的数据格式
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple:
    """
    根据输入图像和目标大小，找到调整大小后输出图像的目标（高度，宽度）维度。

    Args:
        input_image (`np.ndarray`):
            要调整大小的图像。
        size (`int` or `Tuple[int, int]` or List[int] or Tuple[int]):
            用于调整图像大小的尺寸。如果 `size` 是类似 (h, w) 的序列，则输出大小将匹配此大小。

            如果 `size` 是整数且 `default_to_square` 为 `True`，则图像将调整为 (`size`, `size`)。如果 `size` 是整数且 `default_to_square` 为 `False`，则图像的较小边将匹配此数值。即，如果高度 > 宽度，则图像将调整为 (`size * height / width`, `size`)。
        default_to_square (`bool`, *optional*, defaults to `True`):
            当 `size` 是单个整数时如何转换 `size`。如果设置为 `True`，则 `size` 将转换为一个正方形 (`size`, `size`)。如果设置为 `False`，将复制[`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)，支持仅调整最小边并提供可选的 `max_size`。
        max_size (`int`, *optional*):
            调整后的图像的长边允许的最大尺寸：如果图像的长边在根据 `size` 调整大小后大于 `max_size`，则再次调整图像，使得长边等于 `max_size`。因此，可能会覆盖 `size`，即较小的边可能会比 `size` 更短。仅在 `default_to_square` 为 `False` 时使用。
        input_data_format (`ChannelDimension`, *optional*):
            输入图像的通道维度格式。如果未设置，则将使用从输入推断的格式。

    Returns:
        `tuple`: 调整大小后输出图像的目标（高度，宽度）维度。
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
        elif len(size) == 1:
            # 执行与整数大小相同的逻辑
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    # 获取输入图像的尺寸
    height, width = get_image_size(input_image, input_data_format)
    # 确定短边和长边
    short, long = (width, height) if width <= height else (height, width)
    # 请求的新短边尺寸
    requested_new_short = size
    # 计算新的短边和长边的长度，分别为用户请求的新短边长度和根据长短边比例计算的新长边长度
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    # 如果设置了最大尺寸限制
    if max_size is not None:
        # 如果最大尺寸小于或等于用户请求的新短边长度，则引发值错误
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        # 如果新的长边超过了最大尺寸，则重新计算新的短边和长边
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    # 如果宽度小于或等于高度，则返回新的长短边长度组成的元组；否则返回新的短长边长度组成的元组
    return (new_long, new_short) if width <= height else (new_short, new_long)
# 定义一个函数，用于将图像调整大小到指定的大小
def resize(
    image: np.ndarray,
    size: Tuple[int, int],
    resample: "PILImageResampling" = None,  # 重新采样方法，默认为双线性插值
    reducing_gap: Optional[int] = None,     # 优化参数，用于在两步中调整图像大小，默认为None
    data_format: Optional[ChannelDimension] = None,  # 输出图像的通道维度格式，默认为None
    return_numpy: bool = True,              # 是否返回numpy数组，默认为True
    input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道维度格式，默认为None
) -> np.ndarray:                           # 返回值为numpy数组
    """
    使用PIL库将图像调整大小到`(height, width)`指定的大小。

    Args:
        image (`np.ndarray`):
            需要调整大小的图像。
        size (`Tuple[int, int]`):
            调整后的图像尺寸。
        resample (`int`, *可选*, 默认为`PILImageResampling.BILINEAR`):
            用于重新采样的滤波器。
        reducing_gap (`int`, *可选*):
            通过两步调整图像大小来应用优化。较大的`reducing_gap`值，结果越接近公平的重新采样。更多详情请参阅对应的Pillow文档。
        data_format (`ChannelDimension`, *可选*):
            输出图像的通道维度格式。如果未设置，则使用输入图像推断的格式。
        return_numpy (`bool`, *可选*, 默认为`True`):
            是否返回调整大小后的图像作为numpy数组。如果为False，则返回`PIL.Image.Image`对象。
        input_data_format (`ChannelDimension`, *可选*):
            输入图像的通道维度格式。如果未设置，则使用输入图像推断的格式。

    Returns:
        `np.ndarray`: 调整大小后的图像。
    """
    # 检查是否已经加载了vision模块
    requires_backends(resize, ["vision"])

    # 如果未指定重新采样方法，则使用双线性插值
    resample = resample if resample is not None else PILImageResampling.BILINEAR

    # 检查size参数是否有两个元素
    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # 对于所有的变换，我们希望保持与输入图像相同的数据格式，除非另有指定。
    # PIL调整大小后的图像总是通道维度在最后，所以先找到输入格式。
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # 为了与先前的图像特征提取中的调整大小保持向后兼容性，我们使用pillow库来调整图像大小，然后转换回numpy
    do_rescale = False
    # 如果image不是PIL图像，进行适当的缩放
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        # 转换为PIL图像
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    # 获取目标高度和宽度
    height, width = size
    # 调整图像大小
    # PIL图像的格式是(width, height)，因此传入的参数为(width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    # 如果需要返回 numpy 格式的图像
    if return_numpy:
        # 将 PIL 图像转换为 numpy 数组
        resized_image = np.array(resized_image)
        # 如果输入图像的通道维度大小为 1，则在转换为 PIL 图像时会被丢弃，因此需要在必要时添加回来
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # 在从 PIL 图像转换后，图像总是处于通道在最后的格式
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # 如果图像在转换为 PIL 图像之前被重新缩放到 [0, 255] 范围内，则需要将其重新缩放回原始范围
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    # 返回处理后的图像
    return resized_image
# 对图像进行归一化处理，使用给定的均值和标准差
def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    # 获取图像中通道的轴
    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    # 获取图像中通道的数量
    num_channels = image.shape[channel_axis]

    # 将图像转换为 float32 类型以避免在减去 uint8 值时可能出现的错误
    # 如果原始 dtype 是浮点类型，则保留原始 dtype 以防止向上转换为 float16
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    # 如果均值是可迭代对象，则检查其长度是否与通道数量匹配，如果不匹配则引发 ValueError
    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    # 如果标准差是可迭代对象，则检查其长度是否与通道数量匹配，如果不匹配则引发 ValueError
    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    # 根据输入数据格式的不同，对图像进行归一化处理
    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    # 如果指定了输出数据格式，则将图像转换为该格式
    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image


# 对图像进行中心裁剪，以指定的大小进行裁剪。如果图像太小而无法裁剪到给定的大小，则进行填充（因此返回的结果始终为指定的大小）
def center_crop(
    image: np.ndarray,
    size: Tuple[int, int],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).
    """
    Args:
        image (`np.ndarray`):
            The image to crop. 输入的待裁剪图像，类型为 np.ndarray
        size (`Tuple[int, int]`):
            The target size for the cropped image. 裁剪后的目标尺寸，格式为 (height, width)
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image. 输出图像的通道维度格式，可选值有 "channels_first" 或 `ChannelDimension.FIRST` 表示通道在第一维，"channels_last" 或 `ChannelDimension.LAST` 表示通道在最后一维
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image. 输入图像的通道维度格式，可选值有 "channels_first" 或 `ChannelDimension.FIRST` 表示通道在第一维，"channels_last" 或 `ChannelDimension.LAST` 表示通道在最后一维
        return_numpy (`bool`, *optional*):
            Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the
            previous ImageFeatureExtractionMixin method.
                - Unset: will return the same type as the input image.
                - `True`: will return a numpy array.
                - `False`: will return a `PIL.Image.Image` object. 是否返回 numpy 数组格式的裁剪后的图像，用于与之前版本的 ImageFeatureExtractionMixin 方法向后兼容
    Returns:
        `np.ndarray`: The cropped image. 返回裁剪后的图像，类型为 np.ndarray
    """
    requires_backends(center_crop, ["vision"])

    if return_numpy is not None:
        warnings.warn("return_numpy is deprecated and will be removed in v.4.33", FutureWarning)

    return_numpy = True if return_numpy is None else return_numpy

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    # 如果裁剪边界在原始图像范围内，则直接裁剪图像
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        return image

    # 否则，如果图像太小，可能需要填充。好了...
    # 计算新图像的高度和宽度，取裁剪尺寸和原始尺寸的最大值
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    # 创建新图像的形状，将高度和宽度加入到原图像的形状中
    new_shape = image.shape[:-2] + (new_height, new_width)
    # 创建与原图像形状相同的全零数组
    new_image = np.zeros_like(image, shape=new_shape)

    # 如果图像太小，则使用零填充
    # 计算填充边界
    top_pad = (new_height - orig_height) // 2
    bottom_pad = top_pad + orig_height
    left_pad = (new_width - orig_width) // 2
    right_pad = left_pad + orig_width
    # 在新图像上进行填充
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    # 更新裁剪边界，以考虑填充后的图像
    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    # 裁剪填充后的图像，确保裁剪边界在新图像范围内
    new_image = new_image[..., max(0, top): min(new_height, bottom), max(0, left): min(new_width, right)]
    # 调整通道维度的顺序
    new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

    # 如果不返回 NumPy 数组，则将图像转换为 PIL 图像
    if not return_numpy:
        new_image = to_pil_image(new_image)

    # 返回裁剪后的图像
    return new_image
# 将中心格式的边界框转换为角落格式的边界框（torch版本）
def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    # 从中心格式的边界框中解绑出中心点坐标和宽高
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    # 计算角落格式的边界框的坐标
    bbox_corners = torch.stack(
        # 左上角 x, 左上角 y, 右下角 x, 右下角 y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


# 将中心格式的边界框转换为角落格式的边界框（numpy版本）
def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
    # 从中心格式的边界框中解绑出中心点坐标和宽高
    center_x, center_y, width, height = bboxes_center.T
    # 计算角落格式的边界框的坐标
    bboxes_corners = np.stack(
        # 左上角 x, 左上角 y, 右下角 x, 右下角 y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners


# 将中心格式的边界框转换为角落格式的边界框（tensorflow版本）
def _center_to_corners_format_tf(bboxes_center: "tf.Tensor") -> "tf.Tensor":
    # 从中心格式的边界框中解绑出中心点坐标和宽高
    center_x, center_y, width, height = tf.unstack(bboxes_center, axis=-1)
    # 计算角落格式的边界框的坐标
    bboxes_corners = tf.stack(
        # 左上角 x, 左上角 y, 右下角 x, 右下角 y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners


# 以下两个函数受 https://github.com/facebookresearch/detr/blob/master/util/box_ops.py 启发
# 将中心格式的边界框转换为角落格式的边界框（通用版本）
def center_to_corners_format(bboxes_center: TensorType) -> TensorType:
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coodinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    # 在模型前向传播期间使用此函数，尽可能使用输入框架，而不转换为numpy
    if is_torch_tensor(bboxes_center):
        return _center_to_corners_format_torch(bboxes_center)
    elif isinstance(bboxes_center, np.ndarray):
        return _center_to_corners_format_numpy(bboxes_center)
    elif is_tf_tensor(bboxes_center):
        return _center_to_corners_format_tf(bboxes_center)

    raise ValueError(f"Unsupported input type {type(bboxes_center)}")


# 将角落格式的边界框转换为中心格式的边界框（torch版本）
def _corners_to_center_format_torch(bboxes_corners: "torch.Tensor") -> "torch.Tensor":
    # 从角落格式的边界框中解绑出左上角和右下角坐标
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.unbind(-1)
    # 计算中心格式的边界框的坐标
    b = [
        (top_left_x + bottom_right_x) / 2,  # 中心 x
        (top_left_y + bottom_right_y) / 2,  # 中心 y
        (bottom_right_x - top_left_x),  # 宽度
        (bottom_right_y - top_left_y),  # 高度
    ]
    return torch.stack(b, dim=-1)


# 将角落格式的边界框转换为中心格式的边界框（numpy版本）
def _corners_to_center_format_numpy(bboxes_corners: np.ndarray) -> np.ndarray:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
    # 将顶点坐标计算得到的中心点坐标存储到数组中
    bboxes_center = np.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # 计算中心点的 x 坐标
            (top_left_y + bottom_right_y) / 2,  # 计算中心点的 y 坐标
            (bottom_right_x - top_left_x),  # 计算框的宽度
            (bottom_right_y - top_left_y),  # 计算框的高度
        ],
        axis=-1,
    )
    # 返回存储中心点坐标的数组
    return bboxes_center
# 将边界框从角点格式转换为中心格式的函数，输入参数为 TensorFlow 的张量类型，输出也是 TensorFlow 的张量类型
def _corners_to_center_format_tf(bboxes_corners: "tf.Tensor") -> "tf.Tensor":
    # 将输入张量按最后一个维度拆分为四个张量，分别表示左上角、右下角的 x、y 坐标
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = tf.unstack(bboxes_corners, axis=-1)
    # 计算中心坐标和宽高，并重新堆叠成新的张量
    bboxes_center = tf.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # 中心 x 坐标
            (top_left_y + bottom_right_y) / 2,  # 中心 y 坐标
            (bottom_right_x - top_left_x),  # 宽度
            (bottom_right_y - top_left_y),  # 高度
        ],
        axis=-1,
    )
    return bboxes_center


# 将边界框从角点格式转换为中心格式的函数，输入参数为任意类型，输出也是相同的类型
def corners_to_center_format(bboxes_corners: TensorType) -> TensorType:
    """
    Converts bounding boxes from corners format to center format.

    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    center format: contains the coordinate for the center of the box and its the width, height dimensions
        (center_x, center_y, width, height)
    """
    # 如果输入是 PyTorch 张量，则调用对应的转换函数
    if is_torch_tensor(bboxes_corners):
        return _corners_to_center_format_torch(bboxes_corners)
    # 如果输入是 NumPy 数组，则调用对应的转换函数
    elif isinstance(bboxes_corners, np.ndarray):
        return _corners_to_center_format_numpy(bboxes_corners)
    # 如果输入是 TensorFlow 张量，则调用对应的转换函数
    elif is_tf_tensor(bboxes_corners):
        return _corners_to_center_format_tf(bboxes_corners)

    # 如果输入类型不支持，则抛出 ValueError 异常
    raise ValueError(f"Unsupported input type {type(bboxes_corners)}")


# 以下两个函数来自 https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# 版权归 Alexander Kirillov 所有
# 将 RGB 颜色转换为唯一的 ID
def rgb_to_id(color):
    """
    Converts RGB color to unique ID.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


# 将唯一的 ID 转换为 RGB 颜色
def id_to_rgb(id_map):
    """
    Converts unique ID to RGB color.
    """
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


# 表示图像填充模式的枚举类
class PaddingMode(ExplicitEnum):
    """
    Enum class for the different padding modes to use when padding images.
    """

    CONSTANT = "constant"  # 常数填充
    REFLECT = "reflect"  # 反射填充
    REPLICATE = "replicate"  # 复制填充
    SYMMETRIC = "symmetric"  # 对称填充


# 图像填充函数，可指定填充值、填充模式和数据格式
def pad(
    image: np.ndarray,
    padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_values: Union[float, Iterable[float]] = 0.0,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个参数 input_data_format，类型为 Optional[Union[str, ChannelDimension]]，默认值为 None
# 定义一个函数，用于给图像 `image` 添加指定 (height, width) 的 `padding` 和 `mode` 的填充
def pad_image(image: np.ndarray,
              padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
              mode: PaddingMode,
              constant_values: Optional[Union[float, Iterable[float]]] = None,
              data_format: Optional[Union[str, ChannelDimension]] = None,
              input_data_format: Optional[Union[str, ChannelDimension]] = None) -> np.ndarray:
    """
    Pads the `image` with the specified (height, width) `padding` and `mode`.

    Args:
        image (`np.ndarray`):
            The image to pad.
        padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
            Padding to apply to the edges of the height, width axes. Can be one of three formats:
            - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
            - `((before, after),)` yields same before and after pad for height and width.
            - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
        mode (`PaddingMode`):
            The padding mode to use. Can be one of:
                - `"constant"`: pads with a constant value.
                - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                  vector along each axis.
                - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
        constant_values (`float` or `Iterable[float]`, *optional*):
            The value to use for the padding if `mode` is `"constant"`.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.

    Returns:
        `np.ndarray`: The padded image.

    """
    # 如果输入数据格式未指定，则推断输入图像的通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    def _expand_for_data_format(values):
        """
        Convert values to be in the format expected by np.pad based on the data format.
        """
        # 如果值是整数或浮点数，则将其转换为符合 np.pad 预期格式的值
        if isinstance(values, (int, float)):
            values = ((values, values), (values, values))
        # 如果值是单个元组，则将其转换为符合 np.pad 预期格式的值
        elif isinstance(values, tuple) and len(values) == 1:
            values = ((values[0], values[0]), (values[0], values[0]))
        # 如果值是双元组且第一个元素是整数，则保持不变
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
            values = (values, values)
        # 如果值是双元组，则保持不变
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
            values = values
        else:
            raise ValueError(f"Unsupported format: {values}")

        # 根据输入数据格式添加通道维度的填充值
        values = ((0, 0), *values) if input_data_format == ChannelDimension.FIRST else (*values, (0, 0))

        # 如果图像维度为4，则添加额外填充
        values = (0, *values) if image.ndim == 4 else values
        return values

    # 根据数据格式扩展填充
    padding = _expand_for_data_format(padding)

    # 根据填充模式进行填充
    if mode == PaddingMode.CONSTANT:
        constant_values = _expand_for_data_format(constant_values)
        image = np.pad(image, padding, mode="constant", constant_values=constant_values)
    elif mode == PaddingMode.REFLECT:
        image = np.pad(image, padding, mode="reflect")
    elif mode == PaddingMode.REPLICATE:
        image = np.pad(image, padding, mode="edge")
    elif mode == PaddingMode.SYMMETRIC:
        image = np.pad(image, padding, mode="symmetric")
    else:
        raise ValueError(f"Invalid padding mode: {mode}")

    # 根据数据格式将图像转换为通道维度格式
    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image
# TODO (Amy): Accept 1/3/4 channel numpy array as input and return np.array as default
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.

    Args:
        image (Image):
            The image to convert.
    """
    # 检查是否已经安装了相关后端
    requires_backends(convert_to_rgb, ["vision"])

    # 如果输入不是 PIL 图像对象，则直接返回原图
    if not isinstance(image, PIL.Image.Image):
        return image

    # 将图像转换为 RGB 格式
    image = image.convert("RGB")
    return image


def flip_channel_order(
    image: np.ndarray,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Flips the channel order of the image.

    If the image is in RGB format, it will be converted to BGR and vice versa.

    Args:
        image (`np.ndarray`):
            The image to flip.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
    """
    # 推断输入图像的通道维度格式
    input_data_format = infer_channel_dimension_format(image) if input_data_format is None else input_data_format

    # 根据输入图像的通道维度格式进行通道顺序翻转
    if input_data_format == ChannelDimension.LAST:
        # 如果输入图像的通道维度格式为 LAST，则将通道顺序翻转
        image = image[..., ::-1]
    elif input_data_format == ChannelDimension.FIRST:
        # 如果输入图像的通道维度格式为 FIRST，则将通道顺序翻转
        image = image[::-1, ...]
    else:
        # 如果输入图像的通道维度格式不受支持，则引发 ValueError 异常
        raise ValueError(f"Unsupported channel dimension: {input_data_format}")

    # 如果指定了输出图像的通道维度格式，则将图像转换为指定格式
    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    return image
```