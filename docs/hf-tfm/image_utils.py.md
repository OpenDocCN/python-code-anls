# `.\image_utils.py`

```
# 导入必要的库和模块
import base64  # 用于 base64 编解码
import os  # 系统操作相关功能
from io import BytesIO  # 提供字节流操作
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union  # 类型提示相关模块

import numpy as np  # 数组操作库
import requests  # 发送 HTTP 请求的库
from packaging import version  # 版本管理相关功能

from .utils import (  # 导入自定义工具函数
    ExplicitEnum,
    is_jax_tensor,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_vision_available,
    logging,
    requires_backends,
    to_numpy,
)
from .utils.constants import (  # 导入常量
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

# 如果视觉库可用，导入图像处理相关库
if is_vision_available():
    import PIL.Image  # Python Imaging Library，用于图像处理
    import PIL.ImageOps  # PIL 的图像处理操作

    # 根据 PIL 版本选择不同的图像重采样方法
    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
        PILImageResampling = PIL.Image.Resampling
    else:
        PILImageResampling = PIL.Image

# 如果在类型检查模式下，检查是否有 Torch 可用，若可用则导入 Torch
if TYPE_CHECKING:
    if is_torch_available():
        import torch  # PyTorch 深度学习库

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义图像输入类型，可以是 PIL 图像、numpy 数组、Torch 张量的列表
ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]  # noqa


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"  # 通道维度在前
    LAST = "channels_last"  # 通道维度在后


class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"  # COCO 检测注释格式
    COCO_PANOPTIC = "coco_panoptic"  # COCO 全景注释格式


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = AnnotationFormat.COCO_DETECTION.value  # COCO 检测注释格式
    COCO_PANOPTIC = AnnotationFormat.COCO_PANOPTIC.value  # COCO 全景注释格式


AnnotationType = Dict[str, Union[int, str, List[Dict]]]  # 注释类型，字典形式


def is_pil_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image)


def is_valid_image(img):
    return (
        (is_vision_available() and isinstance(img, PIL.Image.Image))  # 图像是 PIL 图像
        or isinstance(img, np.ndarray)  # 图像是 numpy 数组
        or is_torch_tensor(img)  # 图像是 Torch 张量
        or is_tf_tensor(img)  # 图像是 TensorFlow 张量
        or is_jax_tensor(img)  # 图像是 JAX 张量
    )


def valid_images(imgs):
    # 如果是图像列表或元组，则检查每个图像是否有效
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # 如果不是图像列表或元组，则检查单个图像或批量张量是否有效
    elif not is_valid_image(imgs):
        return False
    return True


def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])  # 如果是列表或元组，且第一个元素是有效图像，则认为是批量数据
    return False
# 检查图像是否已经被重新缩放到 [0, 1] 范围内
def is_scaled_image(image: np.ndarray) -> bool:
    if image.dtype == np.uint8:
        return False

    # 可能图像的像素值在 [0, 255] 范围内，但是数据类型是浮点型
    return np.min(image) >= 0 and np.max(image) <= 1


# 确保输入是一个图像列表。如果输入是单个图像，将其转换为长度为 1 的列表。
# 如果输入是图像批次，将其转换为图像列表。
def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    if is_batched(images):
        return images

    # 如果输入是单个 PIL 图像，则创建长度为 1 的列表
    if isinstance(images, PIL.Image.Image):
        # PIL 图像永远不会是批次
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # 图像批次
            images = list(images)
        elif images.ndim == expected_ndims:
            # 单个图像
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, but got"
                f" {images.ndim} dimensions."
            )
        return images
    raise ValueError(
        "Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or "
        f"jax.ndarray, but got {type(images)}."
    )


# 将输入图像转换为 numpy 数组
def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if is_vision_available() and isinstance(img, PIL.Image.Image):
        return np.array(img)
    return to_numpy(img)


# 推断图像的通道维度格式
def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")
    # 检查图像数组的指定维度是否在给定的通道数列表中
    if image.shape[first_dim] in num_channels:
        # 如果第一维度的大小存在于通道数列表中，则返回首维度作为通道维度
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        # 如果最后一维度的大小存在于通道数列表中，则返回末尾维度作为通道维度
        return ChannelDimension.LAST
    # 如果未能确定通道维度的格式，则引发值错误异常
    raise ValueError("Unable to infer channel dimension format")
def get_channel_dimension_axis(
    image: np.ndarray, input_data_format: Optional[Union[ChannelDimension, str]] = None
) -> int:
    """
    Returns the channel dimension axis of the image.

    Args:
        image (`np.ndarray`):
            The image to get the channel dimension axis of.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the image. If `None`, will infer the channel dimension from the image.

    Returns:
        The channel dimension axis of the image.
    """
    # 如果未指定数据格式，从图像推断通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    # 如果数据格式为第一维度优先，则返回倒数第三维度的索引
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    # 如果数据格式为最后一维度优先，则返回倒数第一维度的索引
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    # 抛出异常，不支持的数据格式
    raise ValueError(f"Unsupported data format: {input_data_format}")


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    # 如果未指定通道维度，从图像推断通道维度格式
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    # 如果通道维度为第一维度优先，则返回倒数第二和倒数第一维度的尺寸
    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    # 如果通道维度为最后一维度优先，则返回倒数第三和倒数第二维度的尺寸
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    # 抛出异常，不支持的数据格式
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def is_valid_annotation_coco_detection(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    """
    Checks if the given annotation is a valid COCO detection annotation.

    Args:
        annotation (`Dict[str, Union[List, Tuple]]`):
            The annotation dictionary to validate.

    Returns:
        `True` if the annotation is valid, `False` otherwise.
    """
    # 检查注释是否为有效的 COCO 检测注释
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "annotations" in annotation
        and isinstance(annotation["annotations"], (list, tuple))
        and (
            # 一个图像可能没有注释
            len(annotation["annotations"]) == 0 or isinstance(annotation["annotations"][0], dict)
        )
    ):
        return True
    return False


def is_valid_annotation_coco_panoptic(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    """
    Checks if the given annotation is a valid COCO panoptic segmentation annotation.

    Args:
        annotation (`Dict[str, Union[List, Tuple]]`):
            The annotation dictionary to validate.

    Returns:
        `True` if the annotation is valid, `False` otherwise.
    """
    # 检查注释是否为有效的 COCO 全景分割注释
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "segments_info" in annotation
        and "file_name" in annotation
        and isinstance(annotation["segments_info"], (list, tuple))
        and (
            # 一个图像可能没有分割信息
            len(annotation["segments_info"]) == 0 or isinstance(annotation["segments_info"][0], dict)
        )
    ):
        return True
    return False


def valid_coco_detection_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    """
    Checks if all annotations in the given iterable are valid COCO detection annotations.

    Args:
        annotations (`Iterable[Dict[str, Union[List, Tuple]]]`):
            The iterable of annotations to validate.

    Returns:
        `True` if all annotations are valid, `False` otherwise.
    """
    # 检查给定可迭代对象中的所有注释是否都是有效的 COCO 检测注释
    return all(is_valid_annotation_coco_detection(ann) for ann in annotations)
# 验证所有 COCO Panoptic 注释的有效性
def valid_coco_panoptic_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    # 使用 `is_valid_annotation_coco_panoptic` 函数检查每个注释项是否有效，全部有效则返回 True
    return all(is_valid_annotation_coco_panoptic(ann) for ann in annotations)


def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None) -> "PIL.Image.Image":
    """
    将 `image` 加载为 PIL 图像。

    Args:
        image (`str` or `PIL.Image.Image`):
            要转换为 PIL 图像格式的图像。
        timeout (`float`, *optional*):
            URL 请求的超时值（秒）。

    Returns:
        `PIL.Image.Image`: 一个 PIL 图像。
    """
    # 确保加载图像所需的后端库已加载
    requires_backends(load_image, ["vision"])
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # 如果图像是通过 HTTP 或 HTTPS 协议访问的 URL，则使用 `requests` 获取图像流，并打开为 PIL 图像
            image = PIL.Image.open(requests.get(image, stream=True, timeout=timeout).raw)
        elif os.path.isfile(image):
            # 如果图像路径是一个文件，则直接打开为 PIL 图像
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                # 如果图像以 data:image/ 开头，则取出 base64 编码的部分
                image = image.split(",")[1]

            # 尝试作为 base64 字符串加载图像
            try:
                b64 = base64.b64decode(image, validate=True)
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"图像源格式错误。必须是以 `http://` 或 `https://` 开头的有效 URL，有效的图像文件路径，或者是 base64 编码的字符串。传入值为 {image}。错误信息：{e}"
                )
    elif isinstance(image, PIL.Image.Image):
        # 如果图像已经是 PIL 图像，则保持不变
        image = image
    else:
        raise ValueError(
            "图像格式不正确。应为指向图像的 URL、base64 字符串、本地路径，或者是一个 PIL 图像。"
        )
    # 根据 EXIF 信息对图像进行自动旋转
    image = PIL.ImageOps.exif_transpose(image)
    # 将图像转换为 RGB 模式（如果不是的话）
    image = image.convert("RGB")
    return image


def validate_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[Dict[str, int]] = None,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: Optional["PILImageResampling"] = None,
):
    """
    检查 `ImageProcessor` 的 `preprocess` 方法中常用参数的有效性。
    如果发现参数不兼容，则抛出 `ValueError` 异常。
    许多不兼容性是与模型相关的。`do_pad` 有时需要 `size_divisor`，有时需要 `size_divisibility`，有时需要 `size`。
    新增的模型和处理器应尽量遵循现有参数的使用规则。

    """
    # 如果需要进行重新缩放，并且未指定缩放因子，则抛出数值错误异常
    if do_rescale and rescale_factor is None:
        raise ValueError("rescale_factor must be specified if do_rescale is True.")

    # 如果需要进行填充，并且未指定尺寸可被整除的值，则抛出数值错误异常
    # 在这里，size_divisibility可能被作为size的值传递
    raise ValueError(
        "Depending on model, size_divisibility, size_divisor, pad_size or size must be specified if do_pad is True."
    )

    # 如果需要进行归一化，并且未指定图像均值和标准差，则抛出数值错误异常
    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError("image_mean and image_std must both be specified if do_normalize is True.")

    # 如果需要进行中心裁剪，并且未指定裁剪尺寸，则抛出数值错误异常
    if do_center_crop and crop_size is None:
        raise ValueError("crop_size must be specified if do_center_crop is True.")

    # 如果需要进行调整大小，并且未指定大小或重采样方法，则抛出数值错误异常
    if do_resize and (size is None or resample is None):
        raise ValueError("size and resample must be specified if do_resize is True.")
# 在未来，如果我们有了 TensorFlow 模型，可以在这里添加 TF 的实现。
class ImageFeatureExtractionMixin:
    """
    包含用于准备图像特征的工具函数的 Mixin。
    """

    def _ensure_format_supported(self, image):
        """
        确保图像格式受支持，如果不受支持则引发 ValueError 异常。

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                要检查的图像对象。
        """
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`torch.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        将 `image` 转换为 PIL Image 格式。可选地重新缩放，并在需要时将通道维度放回到最后一个轴。

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                要转换为 PIL Image 格式的图像对象。
            rescale (`bool`, *optional*):
                是否应用缩放因子（使像素值成为介于0到255之间的整数）。如果图像类型是浮点类型，则默认为 `True`。
        """
        self._ensure_format_supported(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # 如果数组是浮点类型，则默认 rescale 为 True。
                rescale = isinstance(image.flat[0], np.floating)
            # 如果通道被移动到第一个维度，我们将其放回到最后。
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def convert_rgb(self, image):
        """
        将 `PIL.Image.Image` 转换为 RGB 格式。

        Args:
            image (`PIL.Image.Image`):
                要转换的图像对象。
        """
        self._ensure_format_supported(image)
        if not isinstance(image, PIL.Image.Image):
            return image

        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
        """
        缩放 numpy 图像按比例 `scale`。

        Args:
            image (`numpy.ndarray`):
                要缩放的图像数组。
            scale (Union[float, int]):
                缩放因子。

        Returns:
            `numpy.ndarray`: 缩放后的图像数组。
        """
        self._ensure_format_supported(image)
        return image * scale
    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
            channel_first (`bool`, *optional*, defaults to `True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        # 确保传入的图像格式受支持
        self._ensure_format_supported(image)

        # 如果图像是 PIL Image 对象，则转换为 numpy 数组
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        # 如果图像是 torch Tensor，则转换为 numpy 数组
        if is_torch_tensor(image):
            image = image.numpy()

        # 如果 rescale 未指定，则根据图像的数据类型判断是否需要进行重新缩放
        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        # 如果需要重新缩放，则将图像像素值缩放到 [0, 1] 范围内
        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        # 如果需要将通道维度放在第一维，则进行维度变换
        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to expand.
        """
        # 确保传入的图像格式受支持
        self._ensure_format_supported(image)

        # 如果图像是 PIL Image 对象，则直接返回，不做任何维度扩展操作
        if isinstance(image, PIL.Image.Image):
            return image

        # 如果图像是 torch Tensor，则在第0维上增加一个维度
        if is_torch_tensor(image):
            image = image.unsqueeze(0)
        else:
            # 如果图像是 numpy 数组，则在第0维上增加一个维度
            image = np.expand_dims(image, axis=0)
        return image
    def normalize(self, image, mean, std, rescale=False):
        """
        Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
        if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to normalize.
            mean (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
            rescale (`bool`, *optional*, defaults to `False`):
                Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
                happen automatically.
        """
        # Ensure that the image format is supported for normalization
        self._ensure_format_supported(image)

        # Convert PIL Image to NumPy array and optionally rescale if required
        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # If image is not PIL, check if rescaling is needed and handle accordingly
        elif rescale:
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            elif is_torch_tensor(image):
                image = self.rescale(image.float(), 1 / 255.0)

        # Ensure mean and std are in the correct format based on image type
        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)

        # Normalize the image based on its dimensions and channel structure
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # RGB or grayscale image
            return (image - mean[:, None, None]) / std[:, None, None]
        else:  # Handle other image types
            return (image - mean) / std

    def flip_channel_order(self, image):
        """
        Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
        `image` to a NumPy array if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
                be first.
        """
        # Ensure that the image format is supported for channel flipping
        self._ensure_format_supported(image)

        # Convert PIL Image to NumPy array for manipulation
        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        # Reverse the order of color channels (RGB <-> BGR)
        return image[::-1, :, :]
    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.
            angle (float or int):
                The rotation angle in degrees. Positive angles are counter-clockwise.
            resample (int, optional):
                An optional resampling filter. Default is `PIL.Image.NEAREST`.
            expand (bool or int, optional):
                Optional expansion flag. If true, the output image size is expanded to contain the entire rotated image.
                If integer, it specifies the desired size of the output image (tuple). Default is 0.
            center (tuple of int, optional):
                Optional center of rotation. Default is None, which means the center is calculated as the center of the image.
            translate (tuple of int, optional):
                Optional translation offset. Default is None.
            fillcolor (tuple or int, optional):
                Optional background color given as a single integer value or a tuple of three integers.

        Returns:
            `PIL.Image.Image`: A rotated `PIL.Image.Image` instance.

        """
        # 如果未指定 resample 参数，则使用默认的 NEAREST 模式
        resample = resample if resample is not None else PIL.Image.NEAREST

        # 确保图像格式受支持，调用对象内部方法进行检查
        self._ensure_format_supported(image)

        # 如果输入的 image 不是 PIL.Image.Image 对象，则将其转换为 PIL.Image.Image 对象
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        # 调用 PIL 库中的 rotate 方法进行图像旋转，并返回旋转后的图像副本
        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )
# 根据给定的注释格式升级注释格式对象
def promote_annotation_format(annotation_format: Union[AnnotionFormat, AnnotationFormat]) -> AnnotationFormat:
    # 当 `AnnotionFormat` 完全废弃后，此行代码可以移除
    return AnnotationFormat(annotation_format.value)


# 验证注释的有效性
def validate_annotations(
    annotation_format: AnnotationFormat,
    supported_annotation_formats: Tuple[AnnotationFormat, ...],
    annotations: List[Dict],
) -> None:
    # 如果注释格式是旧的 `AnnotionFormat` 类型，则发出警告，并升级为 `AnnotationFormat`
    if isinstance(annotation_format, AnnotionFormat):
        logger.warning_once(
            f"`{annotation_format.__class__.__name__}` is deprecated and will be removed in v4.38. "
            f"Please use `{AnnotationFormat.__name__}` instead."
        )
        annotation_format = promote_annotation_format(annotation_format)

    # 检查注释格式是否在支持的注释格式列表中
    if annotation_format not in supported_annotation_formats:
        raise ValueError(f"Unsupported annotation format: {format} must be one of {supported_annotation_formats}")

    # 如果注释格式为 `AnnotationFormat.COCO_DETECTION`，则验证 COCO 检测注释的有效性
    if annotation_format is AnnotationFormat.COCO_DETECTION:
        if not valid_coco_detection_annotations(annotations):
            raise ValueError(
                "Invalid COCO detection annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
                "being a list of annotations in the COCO format."
            )

    # 如果注释格式为 `AnnotationFormat.COCO_PANOPTIC`，则验证 COCO 全景注释的有效性
    if annotation_format is AnnotationFormat.COCO_PANOPTIC:
        if not valid_coco_panoptic_annotations(annotations):
            raise ValueError(
                "Invalid COCO panoptic annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with "
                "the latter being a list of annotations in the COCO format."
            )


# 验证关键字参数的有效性，并发出警告对于未使用或无法识别的关键字参数
def validate_kwargs(valid_processor_keys: List[str], captured_kwargs: List[str]):
    unused_keys = set(captured_kwargs).difference(set(valid_processor_keys))
    if unused_keys:
        unused_key_str = ", ".join(unused_keys)
        # TODO: 这里是否应该发出警告而不是仅仅记录日志？
        logger.warning(f"Unused or unrecognized kwargs: {unused_key_str}.")
```