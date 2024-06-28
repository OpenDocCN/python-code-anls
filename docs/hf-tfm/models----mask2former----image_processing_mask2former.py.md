# `.\models\mask2former\image_processing_mask2former.py`

```
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本进行许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按“原样”分发，不提供任何形式的担保或
# 条件，无论是明示的还是暗示的。参见
# 许可证中的特定语言，管理许可证
# 的权限和限制。

"""Mask2Former 的图像处理器类。"""

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 导入图像处理工具函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
from ...image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
# 导入图像工具函数
from ...image_utils import (
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
# 导入通用工具函数
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 如果 torch 可用，则导入相关模块
if is_torch_available():
    import torch
    from torch import nn

# 从 transformers.models.detr.image_processing_detr.max_across_indices 复制的函数
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值的所有索引的最大值。
    """
    return [max(values_i) for values_i in zip(*values)]

# 从 transformers.models.detr.image_processing_detr.get_max_height_width 复制的函数
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批处理中所有图像的最大高度和宽度。
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        # 如果图像通道维度在前
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        # 如果图像通道维度在后
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)

# 从 transformers.models.detr.image_processing_detr.make_pixel_mask 复制的函数
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    创建像素掩码。
    """
    # 创建图像的像素掩码，其中1表示有效像素，0表示填充像素。
    
    Args:
        image (`np.ndarray`):
            要创建像素掩码的图像。
        output_size (`Tuple[int, int]`):
            掩码的输出尺寸。
    """
    # 获取图像的高度和宽度，根据输入数据格式中的通道维度确定。
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    
    # 创建一个指定输出尺寸的零矩阵作为掩码，数据类型为64位整数。
    mask = np.zeros(output_size, dtype=np.int64)
    
    # 将掩码中图像实际像素部分（未填充部分）置为1。
    mask[:input_height, :input_width] = 1
    
    # 返回生成的像素掩码。
    return mask
# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(segmentation, area_threshold=2):
    """
    Checks the validity of each segment in the segmentation map based on area threshold.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
        area_threshold (`int`, optional):
            Minimum area threshold for valid segments. Defaults to 2.
    Returns:
        `List[int]`: List of valid segment indices.
    """
    segment_ids = torch.unique(segmentation)

    valid_segments = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        area = mask.sum().item()
        if area >= area_threshold:
            valid_segments.append(idx.item())

    return valid_segments
# 将分割地图转换为二进制掩码集合
def convert_segmentation_map_to_binary_masks(
    segmentation_map,
    segments,
    num_classes,
):
    # 初始化二进制掩码列表，每个类别一个列表
    binary_masks = [[] for _ in range(num_classes)]

    # 遍历所有分割对象
    for segment in segments:
        segment_id = segment["id"]
        label_id = segment["label_id"]

        # 创建当前对象的二进制掩码
        binary_mask = (segmentation_map == segment_id).int()
        binary_masks[label_id].append(binary_mask)

    # 将每个类别的掩码列表转换为张量
    binary_masks = [torch.stack(masks) if masks else torch.empty(0, dtype=torch.int32) for masks in binary_masks]

    return binary_masks
    # segmentation_map: "np.ndarray"，声明了一个变量 segmentation_map，类型为 np.ndarray
    # instance_id_to_semantic_id: Optional[Dict[int, int]] = None，声明了一个可选参数 instance_id_to_semantic_id，类型为 Optional[Dict[int, int]]，默认为 None
    # ignore_index: Optional[int] = None，声明了一个可选参数 ignore_index，类型为 Optional[int]，默认为 None
    # reduce_labels: bool = False，声明了一个布尔型参数 reduce_labels，初始化为 False
):
    # 如果要减少标签并且没有提供 ignore_index，则抛出值错误异常
    if reduce_labels and ignore_index is None:
        raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

    # 如果要减少标签，则将标记地图中的所有零值替换为 ignore_index，其他值减一
    if reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # 获取标记地图中的所有唯一标签
    all_labels = np.unique(segmentation_map)

    # 如果指定了 ignore_index，则从所有标签中移除背景标签
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # 为每个对象实例生成二进制掩码
    binary_masks = [(segmentation_map == i) for i in all_labels]
    binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # 如果存在 instance_id_to_semantic_id 映射，则将实例 ID 转换为类别 ID
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        # 遍历所有标签，根据映射表将实例 ID 转换为类别 ID
        for label in all_labels:
            class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
            labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
    else:
        # 否则直接使用所有标签作为类别标签
        labels = all_labels

    # 返回二进制掩码和类别标签，分别转换为 float32 和 int64 类型
    return binary_masks.astype(np.float32), labels.astype(np.int64)


# 从 transformers.models.maskformer.image_processing_maskformer.get_maskformer_resize_output_image_size 复制，并将 maskformer 改为 mask2former
def get_mask2former_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    max_size: Optional[int] = None,
    size_divisor: int = 0,
    default_to_square: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    计算给定所需大小的输出图像大小。

    Args:
        image (`np.ndarray`):
            输入图像。
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            输出图像的大小。
        max_size (`int`, *optional*):
            输出图像的最大大小。
        size_divisor (`int`, *optional*, defaults to 0):
            如果给定了 size_divisor，则输出图像大小将可以被该数整除。
        default_to_square (`bool`, *optional*, defaults to `True`):
            如果未提供大小，是否默认为正方形。
        input_data_format (`ChannelDimension` or `str`, *optional*):
            输入图像的通道维度格式。如果未设置，将使用输入图像的推断格式。

    Returns:
        `Tuple[int, int]`: 输出图像的大小。
    """
    # 调用 get_resize_output_image_size 函数计算输出大小
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )

    # 如果 size_divisor 大于 0，则确保输出大小可以被 size_divisor 整除
    if size_divisor > 0:
        height, width = output_size
        height = int(math.ceil(height / size_divisor) * size_divisor)
        width = int(math.ceil(width / size_divisor) * size_divisor)
        output_size = (height, width)

    return output_size
class Mask2FormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Mask2Former image processor. The image processor can be used to prepare image(s) and optional targets
    for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        size_divisor (`int`, *optional*, defaults to 32):
            Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in
            Swin Transformer.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
            denoted with 0 (background) will be replaced with `ignore_index`.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
            The background label will be replaced by `ignore_index`.



    model_input_names = ["pixel_values", "pixel_mask"]



    定义一个列表 `model_input_names`，包含两个字符串元素 "pixel_values" 和 "pixel_mask"，这些名称用于标识模型的输入。
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
        **kwargs,
    ):
        # 如果传入的参数 kwargs 中包含 "size_divisibility"，发出警告并使用新的参数名称
        if "size_divisibility" in kwargs:
            warnings.warn(
                "The `size_divisibility` argument is deprecated and will be removed in v4.27. Please use "
                "`size_divisor` instead.",
                FutureWarning,
            )
            size_divisor = kwargs.pop("size_divisibility")
        
        # 如果传入的参数 kwargs 中包含 "max_size"，发出警告并设置私有属性 _max_size 为传入的值
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` argument is deprecated and will be removed in v4.27. Please use size['longest_edge']"
                " instead.",
                FutureWarning,
            )
            # 将 max_size 设置为私有属性，以便在预处理方法中将其作为默认值传递，同时仍然可以将 size 作为整数传递
            self._max_size = kwargs.pop("max_size")
        else:
            # 否则设置默认的 _max_size 值为 1333
            self._max_size = 1333
        
        # 如果 size 参数为 None，则设置默认的 size 字典，包含 "shortest_edge" 和 "longest_edge" 键
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        # 根据传入的 size 和 _max_size 参数，获取最终的 size 字典
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        # 调用父类的初始化方法，传入其余的关键字参数
        super().__init__(**kwargs)

        # 初始化对象的各个属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels
        # 设置一个包含有效处理器键的列表，用于后续处理
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "instance_id_to_semantic_id",
            "do_resize",
            "size",
            "size_divisor",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "ignore_index",
            "reduce_labels",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    # 重写了基类中的 `from_dict` 方法，用于确保当通过 `from_dict` 和 kwargs 创建图像处理器时参数可以更新，
    # 例如 `Mask2FormerImageProcessor.from_pretrained(checkpoint, max_size=800)`
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        # 复制输入的图像处理器字典，以免修改原始输入
        image_processor_dict = image_processor_dict.copy()
        # 如果 `kwargs` 中包含 "max_size"，则更新图像处理器字典中的 "max_size" 参数，并从 `kwargs` 中删除
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果 `kwargs` 中包含 "size_divisibility"，则更新图像处理器字典中的 "size_divisibility" 参数，并从 `kwargs` 中删除
        if "size_divisibility" in kwargs:
            image_processor_dict["size_divisibility"] = kwargs.pop("size_divisibility")
        # 调用基类的 `from_dict` 方法，传递更新后的图像处理器字典和剩余的 `kwargs`
        return super().from_dict(image_processor_dict, **kwargs)

    # 从 transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.resize 处复制，
    # 将函数名从 `get_maskformer_resize_output_image_size` 修改为 `get_mask2former_resize_output_image_size`
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 0,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
            image: np.ndarray,
            size: Dict[str, int],
            size_divisor: int = 0,
            resample: PILImageResampling = PILImageResampling.BILINEAR,
            data_format: Optional[Union[ChannelDimension, str]] = None,
            input_data_format: Optional[Union[ChannelDimension, str]] = None,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
    
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                The size of the output image.
            size_divisor (`int`, *optional*, defaults to 0):
                If `size_divisor` is given, the output image size will be divisible by the number.
            resample (`PILImageResampling` resampling filter, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    
        # Check if deprecated `max_size` parameter is present and issue a warning
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
    
        # Adjust `size` using utility function `get_size_dict`
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
    
        # Determine whether `size` contains `shortest_edge` and `longest_edge` or `height` and `width`
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            # Raise ValueError if `size` does not contain necessary keys
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
    
        # Adjust `size` using another utility function `get_mask2former_resize_output_image_size`
        size = get_mask2former_resize_output_image_size(
            image=image,
            size=size,
            max_size=max_size,
            size_divisor=size_divisor,
            default_to_square=False,
            input_data_format=input_data_format,
        )
    
        # Resize the `image` using specified parameters and return the resized image
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image
    
    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.convert_segmentation_map_to_binary_masks
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        """
        Convert a segmentation map to binary masks based on instance IDs and semantic IDs.

        Args:
            segmentation_map (`np.ndarray`):
                The input segmentation map to be converted.
            instance_id_to_semantic_id (Optional[Dict[int, int]], *optional*):
                Mapping from instance IDs to semantic IDs.
            ignore_index (Optional[int], *optional*):
                Index to ignore in the segmentation map.
            reduce_labels (`bool`, *optional*):
                Whether to reduce the number of unique labels in the masks.

        Returns:
            `np.ndarray`: Binary masks corresponding to the segmentation map.
        """
        reduce_labels = reduce_labels if reduce_labels is not None else self.reduce_labels
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            reduce_labels=reduce_labels,
        )

    def __call__(self, images, segmentation_maps=None, **kwargs) -> BatchFeature:
        """
        Perform preprocessing on images and segmentation maps.

        Args:
            images (ImageInput):
                Input images to preprocess.
            segmentation_maps (Optional[np.ndarray], *optional*):
                Optional segmentation maps corresponding to the images.
            **kwargs:
                Additional keyword arguments for preprocessing.

        Returns:
            BatchFeature: Preprocessed batch of images and segmentation maps.
        """
        return self.preprocess(images, segmentation_maps=segmentation_maps, **kwargs)

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 如果需要进行图片尺寸调整，则调用 resize 方法
        if do_resize:
            image = self.resize(
                image, size=size, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format
            )
        # 如果需要进行图片缩放，则调用 rescale 方法
        if do_rescale:
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        # 如果需要进行图片标准化，则调用 normalize 方法
        if do_normalize:
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # 返回预处理后的图片数据
        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 将输入的图像转换为 numpy 数组
        image = to_numpy_array(image)
        # 如果图像已经是缩放过的并且需要进行再次缩放，则记录警告信息
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 推断输入数据的通道格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # 调用 _preprocess 方法进行实际的图像预处理
        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            size_divisor=size_divisor,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        # 如果指定了输出数据的通道格式，则进行格式转换
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回预处理后的图像数据
        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = 0,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        # 将分割地图转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        
        # 如果分割地图的维度是 2，则添加一个通道维度，这在某些转换中是必要的
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]  # 在第一维度上添加一个维度
            input_data_format = ChannelDimension.FIRST  # 设置数据格式为第一通道优先
        else:
            added_channel_dim = False
            if input_data_format is None:
                # 推断通道维度的格式
                input_data_format = infer_channel_dimension_format(segmentation_map)
        
        # TODO: (Amy)
        # 重构分割地图处理流程，包括减少标签和调整大小，以便不丢弃大于 255 的段ID。
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            size_divisor=size_divisor,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )
        
        # 如果为了处理而添加了额外的通道维度，则移除它
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        
        # 返回预处理后的分割地图
        return segmentation_map

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        size_divisor: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        # 该方法未提供完整代码，仅作为示例代码的一部分，后续部分应继续补充

    # 以下是从 transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image 复制的
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
        # 获取输入图像的高度和宽度，根据输入数据格式确定通道维度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出的目标高度和宽度
        output_height, output_width = output_size

        # 计算需要填充的底部和右侧像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 定义填充的方式，上部和左部都不需要填充，底部和右侧填充零值
        padding = ((0, pad_bottom), (0, pad_right))
        # 对图像进行填充操作，使用常量值填充，可以指定数据格式和输入数据格式
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

    # 从 transformers.models.vilt.image_processing_vilt.ViltImageProcessor.pad 复制过来的函数
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
        # 获取批量图像中的最大高度和宽度，用于填充大小的确定
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对每张图像进行填充操作，使它们的尺寸都扩展到批量中最大的高度和宽度
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
        # 将填充后的图像数据放入字典中
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 生成每张图像的像素掩码，并放入列表中
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 将像素掩码列表放入数据字典中
            data["pixel_mask"] = masks

        # 返回填充后的批量特征对象，包括数据字典和返回的张量类型
        return BatchFeature(data=data, tensor_type=return_tensors)

    def encode_inputs(
        self,
        pixel_values_list: List[ImageInput],
        segmentation_maps: ImageInput = None,
        instance_id_to_semantic_id: Optional[Union[List[Dict[int, int]], Dict[int, int]]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Encodes input images and optional segmentation maps into a format suitable for model input.

        Args:
            pixel_values_list (`List[ImageInput]`):
                List of input images to encode.
            segmentation_maps (`ImageInput`, *optional*):
                Optional segmentation maps associated with input images.
            instance_id_to_semantic_id (`Optional[Union[List[Dict[int, int]], Dict[int, int]]]`, *optional*):
                Mapping from instance IDs to semantic IDs, if applicable.
            ignore_index (`Optional[int]`, *optional*):
                Index to ignore in the encoding process.
            reduce_labels (`bool`, *optional*, defaults to `False`):
                Whether to reduce the number of unique labels in the segmentation maps.
            return_tensors (`Optional[Union[str, TensorType]]`, *optional*):
                The type of tensors to return.
            input_data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The channel dimension format of the input images.

        Returns:
            Encoded inputs suitable for model processing.
        """
        # 编码输入图像及其相关信息，返回适合模型输入的格式
        ...

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Post-processes semantic segmentation model outputs.

        Args:
            outputs: Model outputs to be post-processed.
            target_sizes (`Optional[List[Tuple[int, int]]]`, *optional*):
                Target sizes for resizing outputs.

        Returns:
            Post-processed outputs.
        """
        # 对语义分割模型的输出进行后处理
        ...
    ) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        # Extract logits for class queries and mask queries from model outputs
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale masks_queries_logits back to preprocessed image size (384, 384)
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class from class_queries_logits (`[..., :-1]`)
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Compute semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps if target_sizes is provided
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # If target_sizes is None, compute semantic segmentation maps for each batch item
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
    # 对实例分割模型输出进行后处理，包括阈值化、合并重叠的实例掩码区域等
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 置信度阈值，用于筛选实例
        mask_threshold: float = 0.5,  # 掩码阈值，用于阈值化掩码
        overlap_mask_area_threshold: float = 0.8,  # 重叠掩码区域的阈值，用于合并重叠实例的掩码
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标大小列表，用于尺寸调整
        return_coco_annotation: Optional[bool] = False,  # 是否返回 COCO 格式的注释
        return_binary_maps: Optional[bool] = False,  # 是否返回二进制地图
    ):
    
    # 对全景分割模型输出进行后处理，包括阈值化、合并重叠的分割标签等
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 置信度阈值，用于筛选分割标签
        mask_threshold: float = 0.5,  # 掩码阈值，用于阈值化掩码
        overlap_mask_area_threshold: float = 0.8,  # 重叠掩码区域的阈值，用于合并重叠分割标签的掩码
        label_ids_to_fuse: Optional[Set[int]] = None,  # 要融合的标签 ID 集合
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标大小列表，用于尺寸调整
    ):
```