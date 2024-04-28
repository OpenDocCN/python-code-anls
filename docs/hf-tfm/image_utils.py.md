# `.\transformers\image_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version

# 导入自定义工具函数和常量
from .utils import (
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
from .utils.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

# 如果视觉库可用，则导入 PIL.Image 和 PIL.ImageOps
if is_vision_available():
    import PIL.Image
    import PIL.ImageOps

    # 根据 PIL 版本选择不同的重采样方法
    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
        PILImageResampling = PIL.Image.Resampling
    else:
        PILImageResampling = PIL.Image

# 如果是类型检查，则导入 torch 库
if TYPE_CHECKING:
    if is_torch_available():
        import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义图像输入类型
ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]

# 定义通道维度枚举类
class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"

# 定义注释格式枚举类
class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"

# 定义注释格式别名
class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = AnnotationFormat.COCO_DETECTION.value
    COCO_PANOPTIC = AnnotationFormat.COCO_PANOPTIC.value

# 定义注释类型
AnnotationType = Dict[str, Union[int, str, List[Dict]]]

# 判断是否为 PIL 图像
def is_pil_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image)

# 判断是否为有效图像
def is_valid_image(img):
    return (
        (is_vision_available() and isinstance(img, PIL.Image.Image))
        or isinstance(img, np.ndarray)
        or is_torch_tensor(img)
        or is_tf_tensor(img)
        or is_jax_tensor(img)
    )

# 检查图像是否有效
def valid_images(imgs):
    # 如果是图像列表，则检查每个图像是否有效
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # 如果不是图像列表，则检查单个图像或批量张量图像是否有效
    elif not is_valid_image(imgs):
        return False
    return True

# 判断是否为批量图像
def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False
def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    # 检查像素值是否已经被重新缩放到 [0, 1] 范围内
    if image.dtype == np.uint8:
        return False

    # 可能图像的像素值在 [0, 255] 范围内，但是是浮点类型
    return np.min(image) >= 0 and np.max(image) <= 1


def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    """
    Ensure that the input is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """
    # 检查输入是否为批量图像
    if is_batched(images):
        return images

    # 如果输入是单个图像，则将其转换为长度为1的列表
    if isinstance(images, PIL.Image.Image):
        # PIL 图像永远不会是批量的
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # 图像批量
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


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if is_vision_available() and isinstance(img, PIL.Image.Image):
        return np.array(img)
    return to_numpy(img)


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
    # 检查图像的第一个维度是否在通道数列表中
    if image.shape[first_dim] in num_channels:
        # 如果是，则返回通道维度为第一个维度
        return ChannelDimension.FIRST
    # 如果第一个维度不在通道数列表中，则检查最后一个维度是否在通道数列表中
    elif image.shape[last_dim] in num_channels:
        # 如果是，则返回通道维度为最后一个维度
        return ChannelDimension.LAST
    # 如果都不在通道数列表中，则抛出数值错误异常
    raise ValueError("Unable to infer channel dimension format")
# 获取图像的通道维度轴
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
    # 如果未指定输入数据格式，则从图像中推断通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    # 如果通道维度格式为第一个维度
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    # 如果通道维度格式为最后一个维度
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    # 抛出异常，表示不支持的数据格式
    raise ValueError(f"Unsupported data format: {input_data_format}")


# 获取图像的高度和宽度
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
    # 如果未指定通道维度，则从图像中推断通道维度格式
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    # 如果通道维度在第一个维度
    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    # 如果通道维度在最后一个维度
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        # 抛出异常，表示不支持的数据格式
        raise ValueError(f"Unsupported data format: {channel_dim}")


# 检查 COCO 检测注释是否有效
def is_valid_annotation_coco_detection(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "annotations" in annotation
        and isinstance(annotation["annotations"], (list, tuple))
        and (
            # 一张图像可以没有注释
            len(annotation["annotations"]) == 0 or isinstance(annotation["annotations"][0], dict)
        )
    ):
        return True
    return False


# 检查 COCO 全景分割注释是否有效
def is_valid_annotation_coco_panoptic(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "segments_info" in annotation
        and "file_name" in annotation
        and isinstance(annotation["segments_info"], (list, tuple))
        and (
            # 一张图像可以没有分割信息
            len(annotation["segments_info"]) == 0 or isinstance(annotation["segments_info"][0], dict)
        )
    ):
        return True
    return False


# 检查 COCO 检测注释是否有效的批量操作
def valid_coco_detection_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    # 检查所有注释是否有效
    return all(is_valid_annotation_coco_detection(ann) for ann in annotations)
# 检查给定的 COCO Panoptic 标注是否有效，返回是否所有标注都有效
def valid_coco_panoptic_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all(is_valid_annotation_coco_panoptic(ann) for ann in annotations)


# 加载图像并转换为 PIL Image 格式
def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    # 检查是否已加载 "vision" 后端
    requires_backends(load_image, ["vision"])
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # 实际检查协议，以便使用本地文件
            image = PIL.Image.open(requests.get(image, stream=True, timeout=timeout).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # 尝试以 base64 加载
            try:
                b64 = base64.b64decode(image, validate=True)
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    # 修正图像的方向
    image = PIL.ImageOps.exif_transpose(image)
    # 转换图像为 RGB 格式
    image = image.convert("RGB")
    return image


# 在未来，可以在此处添加 TF 实现，当有 TF 模型时
class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """

    # 确保图像格式受支持
    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`torch.Tensor` are."
            )
    def to_pil_image(self, image, rescale=None):
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        # 确保图像格式受支持
        self._ensure_format_supported(image)

        # 如果是 torch.Tensor 类型的图像，转换为 numpy 数组
        if is_torch_tensor(image):
            image = image.numpy()

        # 如果是 numpy 数组
        if isinstance(image, np.ndarray):
            if rescale is None:
                # 如果未指定 rescale 参数，默认为数组是浮点类型
                rescale = isinstance(image.flat[0], np.floating)
            # 如果通道维度已经移动到第一个维度，将其放回到最后一个维度
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def convert_rgb(self, image):
        """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
        # 确保图像格式受支持
        self._ensure_format_supported(image)
        # 如果不是 PIL.Image.Image 类型的图像，直接返回
        if not isinstance(image, PIL.Image.Image):
            return image

        # 转换为 RGB 格式
        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
        """
        Rescale a numpy image by scale amount
        """
        # 确保图像格式受支持
        self._ensure_format_supported(image)
        # 返回按比例缩放后的 numpy 图像
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
        # 确保图像格式受支持
        self._ensure_format_supported(image)

        # 如果图像是 PIL 图像，则转换为 NumPy 数组
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        # 如果是 torch 张量，则转换为 NumPy 数组
        if is_torch_tensor(image):
            image = image.numpy()

        # 如果未指定 rescale 参数，则根据图像的第一个像素值类型来确定是否需要缩放
        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        # 如果需要缩放，则将图像转换为浮点数并进行缩放
        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        # 如果需要将通道维度放在第一维，并且图像维度为 3，则转置图像维度
        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        # 返回转换后的图像数组
        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to expand.
        """
        # 确保图像格式受支持
        self._ensure_format_supported(image)

        # 如果是 PIL 图像，则不做任何处理，直接返回
        if isinstance(image, PIL.Image.Image):
            return image

        # 如果是 torch 张量，则在第一维度上增加一个维度
        if is_torch_tensor(image):
            image = image.unsqueeze(0)
        else:
            # 如果是 NumPy 数组，则在第一维度上增加一个维度
            image = np.expand_dims(image, axis=0)
        # 返回处理后的图像
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
        # 确保输入的图像格式受支持
        self._ensure_format_supported(image)

        # 如果输入的图像是 PIL 图像，则将其转换为 NumPy 数组并进行重新缩放
        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # 如果输入的图像不是 PIL 图像，但需要重新缩放
        elif rescale:
            # 如果输入的图像是 NumPy 数组
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            # 如果输入的图像是 Torch 张量
            elif is_torch_tensor(image):
                image = self.rescale(image.float(), 1 / 255.0)

        # 如果输入的图像是 NumPy 数组
        if isinstance(image, np.ndarray):
            # 如果均值不是 NumPy 数组，则转换为与图像数据类型相同的 NumPy 数组
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            # 如果标准差不是 NumPy 数组，则转换为与图像数据类型相同的 NumPy 数组
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        # 如果输入的图像是 Torch 张量
        elif is_torch_tensor(image):
            import torch

            # 如果均值不是 Torch 张量，则转换为 Torch 张量
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            # 如果标准差不是 Torch 张量，则转换为 Torch 张量
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)

        # 如果图像是三维的且第一个维度是1或3
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
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
        # 确保输入的图像格式受支持
        self._ensure_format_supported(image)

        # 如果输入的图像是 PIL 图像，则将其转换为 NumPy 数组
        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        # 返回颜色通道顺序翻转后的图像
        return image[::-1, :, :]
    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.

        Returns:
            image: A rotated `PIL.Image.Image`.
        """
        # 设置默认的重采样方法为最近邻插值
        resample = resample if resample is not None else PIL.Image.NEAREST

        # 确保图像格式受支持
        self._ensure_format_supported(image)

        # 如果图像不是 PIL.Image.Image 类型，则转换为该类型
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        # 返回旋转后的图像副本
        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )
# 定义一个函数，用于提升注释格式，接受一个注释格式参数，返回一个注释格式对象
def promote_annotation_format(annotation_format: Union[AnnotionFormat, AnnotationFormat]) -> AnnotationFormat:
    # 当 `AnnotionFormat` 完全废弃时，此处可以移除
    return AnnotationFormat(annotation_format.value)

# 定义一个函数，用于验证注释
def validate_annotations(
    annotation_format: AnnotationFormat,
    supported_annotation_formats: Tuple[AnnotationFormat, ...],
    annotations: List[Dict],
) -> None:
    # 如果注释格式是 `AnnotionFormat` 类型的，则发出警告
    if isinstance(annotation_format, AnnotionFormat):
        logger.warning_once(
            f"`{annotation_format.__class__.__name__}` is deprecated and will be removed in v4.38. "
            f"Please use `{AnnotationFormat.__name__}` instead."
        )
        # 提升注释格式
        annotation_format = promote_annotation_format(annotation_format)

    # 如果注释格式不在支持的注释格式列表中，则抛出值错误
    if annotation_format not in supported_annotation_formats:
        raise ValueError(f"Unsupported annotation format: {format} must be one of {supported_annotation_formats}")

    # 如果注释格式为 COCO_DETECTION，则验证 COCO 检测注释是否有效
    if annotation_format is AnnotationFormat.COCO_DETECTION:
        if not valid_coco_detection_annotations(annotations):
            raise ValueError(
                "Invalid COCO detection annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
                "being a list of annotations in the COCO format."
            )

    # 如果注释格式为 COCO_PANOPTIC，则验证 COCO 全景注释是否有效
    if annotation_format is AnnotationFormat.COCO_PANOPTIC:
        if not valid_coco_panoptic_annotations(annotations):
            raise ValueError(
                "Invalid COCO panoptic annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with "
                "the latter being a list of annotations in the COCO format."
            )
```