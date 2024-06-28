# `.\models\maskformer\image_processing_maskformer.py`

```
# coding=utf-8
# 版权 2022 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何明示或暗示的担保或条件。
# 请参阅许可证获取特定语言的权限和限制。
"""MaskFormer 的图像处理器类。"""

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
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
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果是类型检查环境
if TYPE_CHECKING:
    from transformers import MaskFormerForInstanceSegmentationOutput

# 如果 Torch 可用
if is_torch_available():
    import torch
    from torch import nn

# 从 transformers.models.detr.image_processing_detr.max_across_indices 复制过来
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值的所有索引中的最大值。
    """
    return [max(values_i) for values_i in zip(*values)]

# 从 transformers.models.detr.image_processing_detr.get_max_height_width 复制过来
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批次中所有图像的最大高度和宽度。
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)

# 从 transformers.models.detr.image_processing_detr.make_pixel_mask 复制过来
def make_pixel_mask(
    # image: np.ndarray 是一个参数，表示输入的图像数据，类型为 numpy 的多维数组
    # output_size: Tuple[int, int] 是一个参数，表示输出图像的尺寸，以元组形式给出，包含两个整数值
    # input_data_format: Optional[Union[str, ChannelDimension]] = None 是一个可选参数，用于指定输入数据的格式，可以是字符串或 ChannelDimension 类型的联合类型，如果不提供则默认为 None
# 根据输入图像创建像素掩码，其中1表示有效像素，0表示填充像素。
def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
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
    # 创建一个全零的掩码，形状为output_size，数据类型为np.int64
    mask = np.zeros(output_size, dtype=np.int64)
    # 将掩码的有效像素部分设为1，根据输入图像的实际大小进行裁剪
    mask[:input_height, :input_width] = 1
    return mask


# 从transformers.models.detr.image_processing_detr.binary_mask_to_rle复制而来
def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    # 如果mask是torch.Tensor，则转换为numpy数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将二进制掩码展平为一维数组
    pixels = mask.flatten()
    # 在数组两端各加一个0，以确保算法正确性
    pixels = np.concatenate([[0], pixels, [0]])
    # 计算像素值改变的位置索引，构建运行长度编码
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# 从transformers.models.detr.image_processing_detr.convert_segmentation_to_rle复制而来
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    # 获取分割图中所有唯一的标签值
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    # 遍历每个唯一的标签值
    for idx in segment_ids:
        # 创建与当前标签匹配的二进制掩码
        mask = torch.where(segmentation == idx, 1, 0)
        # 将二进制掩码转换为运行长度编码（RLE）
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


# 从transformers.models.detr.image_processing_detr.remove_low_and_no_objects复制而来
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    """
    # 确保所有输入张量的第一个维度大小相同
    # 检查输入的`masks`、`scores`和`labels`张量是否具有相同的第一个维度
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        # 如果它们的第一个维度不同，抛出数值错误异常
        raise ValueError("mask, scores and labels must have the same shape!")

    # 创建布尔张量 `to_keep`，其中元素为真（True）的条件是标签不等于 `num_labels` 并且得分大于 `object_mask_threshold`
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    # 返回根据 `to_keep` 布尔张量过滤后的 `masks`、`scores` 和 `labels`
    return masks[to_keep], scores[to_keep], labels[to_keep]
# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第 k 类相关的掩码
    mask_k = mask_labels == k
    # 计算第 k 类掩码的总面积
    mask_k_area = mask_k.sum()

    # 计算预测概率中第 k 类的总面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 检查是否存在有效掩码
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除断开的小段
    if mask_exists:
        # 计算掩码面积比例
        area_ratio = mask_k_area / original_area
        # 如果面积比例不大于重叠掩码面积阈值，则认为不存在有效掩码
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 创建一个全零的分割图像，用于存储每个像素的分割标签
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    # 用于存储检测到的所有分割结果
    segments: List[Dict] = []

    if target_size is not None:
        # 如果有指定目标尺寸，则插值调整掩码概率张量的尺寸
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # 将每个掩码乘以其预测分数
    mask_probs *= pred_scores.view(-1, 1, 1)
    # 获取每个像素位置上概率最大的类别标签作为分割标签
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 用于记录每个类别的实例数量
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查是否存在有效的分割掩码并且足够大
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象的分割标签添加到最终的分割图像中
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            # 将当前对象的分割信息添加到分割列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


# TODO: (Amy) Move to image_transforms
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
    # segmentation_map 是一个变量，用来表示分割地图，类型为 numpy 数组（np.ndarray）

    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    # instance_id_to_semantic_id 是一个可选的字典类型变量，用来映射实例 ID 到语义 ID，键和值都是整数类型

    ignore_index: Optional[int] = None,
    # ignore_index 是一个可选的整数变量，用来指定在分割过程中忽略的索引值，默认为 None

    reduce_labels: bool = False,
    # reduce_labels 是一个布尔型变量，用来表示是否需要减少标签的数量，默认为 False
    ):
        # 如果 reduce_labels 为 True 但 ignore_index 未提供，则抛出数值错误异常
        raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

    if reduce_labels:
        # 如果 reduce_labels 为 True，则将 segmentation_map 中值为 0 的位置替换为 ignore_index，其它位置减去 1
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # 获取唯一的标签 ids（基于输入是类别还是实例）
    all_labels = np.unique(segmentation_map)

    # 如果 ignore_index 不为空，则删除背景标签
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # 为每个对象实例生成二进制掩码
    binary_masks = [(segmentation_map == i) for i in all_labels]
    binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # 如果 instance_id_to_semantic_id 不为空，则将实例 ids 转换为类别 ids
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            # 根据 reduce_labels 来选择是否需要对 label 进行调整
            class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
            labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
    else:
        labels = all_labels

    # 返回二进制掩码数组和标签数组，掩码为浮点数类型，标签为整数类型
    return binary_masks.astype(np.float32), labels.astype(np.int64)


def get_maskformer_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    max_size: Optional[int] = None,
    size_divisor: int = 0,
    default_to_square: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    根据所需大小计算输出图像的大小。

    Args:
        image (`np.ndarray`):
            输入图像。
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            输出图像的大小。
        max_size (`int`, *可选*):
            输出图像的最大大小。
        size_divisor (`int`, *可选*, 默认为 0):
            如果提供了 `size_divisor`，输出图像大小将可以被此数整除。
        default_to_square (`bool`, *可选*, 默认为 `True`):
            如果未提供大小是否默认为正方形。
        input_data_format (`ChannelDimension` or `str`, *可选*):
            输入图像的通道维度格式。如果未设置，则使用输入的推断格式。

    Returns:
        `Tuple[int, int]`: 输出图像的大小。
    """
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )

    if size_divisor > 0:
        height, width = output_size
        height = int(math.ceil(height / size_divisor) * size_divisor)
        width = int(math.ceil(width / size_divisor) * size_divisor)
        output_size = (height, width)

    # 返回计算后的输出图像大小
    return output_size


class MaskFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MaskFormer image processor. The image processor can be used to prepare image(s) and optional targets
    for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

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
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
            The background label will be replaced by `ignore_index`.

    """
    # 定义模型输入的名称列表，包含像素值和像素掩码
    model_input_names = ["pixel_values", "pixel_mask"]
    
    # 初始化函数，用于创建和配置数据预处理类的实例
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，包含宽度和高度
        size_divisor: int = 32,  # 图像大小调整的除数，用于确保大小是32的倍数
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整时使用的插值方法，默认为双线性插值
        do_rescale: bool = True,  # 是否进行图像像素值的缩放，默认为True
        rescale_factor: float = 1 / 255,  # 图像像素值缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像像素值的归一化，默认为True
        image_mean: Union[float, List[float]] = None,  # 图像归一化时的均值，可以是单个值或列表
        image_std: Union[float, List[float]] = None,  # 图像归一化时的标准差，可以是单个值或列表
        ignore_index: Optional[int] = None,  # 可选参数，指定要忽略的标签索引
        do_reduce_labels: bool = False,  # 是否减少标签数量，默认为False
        **kwargs,  # 其他可能的关键字参数，灵活配置
    ):
        ):
            # 检查是否传入了 `size_divisibility` 参数，如果有则发出警告并使用 `size_divisor` 替代
            if "size_divisibility" in kwargs:
                warnings.warn(
                    "The `size_divisibility` argument is deprecated and will be removed in v4.27. Please use "
                    "`size_divisor` instead.",
                    FutureWarning,
                )
                size_divisor = kwargs.pop("size_divisibility")

            # 检查是否传入了 `max_size` 参数，如果有则发出警告并将其作为私有属性 `_max_size` 存储
            if "max_size" in kwargs:
                warnings.warn(
                    "The `max_size` argument is deprecated and will be removed in v4.27. Please use size['longest_edge']"
                    " instead.",
                    FutureWarning,
                )
                # 将 `max_size` 作为默认值传递给 `preprocess` 方法的私有属性 `_max_size`
                self._max_size = kwargs.pop("max_size")
            else:
                # 如果未传入 `max_size` 参数，默认设为 1333
                self._max_size = 1333

            # 检查是否传入了 `reduce_labels` 参数，如果有则发出警告并使用 `do_reduce_labels` 替代
            if "reduce_labels" in kwargs:
                warnings.warn(
                    "The `reduce_labels` argument is deprecated and will be removed in v4.27. Please use "
                    "`do_reduce_labels` instead.",
                    FutureWarning,
                )
                do_reduce_labels = kwargs.pop("reduce_labels")

            # 如果未指定 `size` 参数，则设置默认的 `size` 字典，包括 `shortest_edge` 和 `longest_edge`
            size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
            # 获取处理后的 `size` 字典，确保不超过 `max_size` 的限制
            size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

            # 调用父类的初始化方法，传入所有的关键字参数
            super().__init__(**kwargs)

            # 初始化对象的各种属性
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
            self.do_reduce_labels = do_reduce_labels

            # 定义有效的处理器关键字列表，用于后续处理器方法的调用和验证
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
                "do_reduce_labels",
                "return_tensors",
                "data_format",
                "input_data_format",
            ]

        @classmethod
    # 重写基类的 `from_dict` 方法，用于从字典创建图像处理器对象，并确保参数更新
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `MaskFormerImageProcessor.from_pretrained(checkpoint, max_size=800)`
        """
        # 复制输入的字典，以防修改原始输入
        image_processor_dict = image_processor_dict.copy()
        
        # 如果 `kwargs` 中包含 `max_size` 参数，则更新到 `image_processor_dict` 中，并从 `kwargs` 中删除
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        
        # 如果 `kwargs` 中包含 `size_divisibility` 参数，则更新到 `image_processor_dict` 中，并从 `kwargs` 中删除
        if "size_divisibility" in kwargs:
            image_processor_dict["size_divisibility"] = kwargs.pop("size_divisibility")
        
        # 调用基类的 `from_dict` 方法，使用更新后的 `image_processor_dict` 和其余 `kwargs` 创建图像处理器对象
        return super().from_dict(image_processor_dict, **kwargs)

    # 图像调整大小的方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 0,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
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
        # Check if the deprecated `max_size` parameter is used and issue a warning
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # Transform `size` into a standardized dictionary format
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        # Handle different formats of `size` and set `size` and `max_size` accordingly
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            # Raise an error if `size` does not contain expected keys
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # Compute the output image size after resizing
        size = get_maskformer_resize_output_image_size(
            image=image,
            size=size,
            max_size=max_size,
            size_divisor=size_divisor,
            default_to_square=False,
            input_data_format=input_data_format,
        )
        # Resize the input `image` using specified parameters
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        # Return the resized image
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

    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        """
        Convert a segmentation map to binary masks.

        Args:
            segmentation_map (`np.ndarray`):
                The input segmentation map.
            instance_id_to_semantic_id (Optional[Dict[int, int]]):
                Mapping from instance IDs to semantic IDs. If not provided, no mapping is applied.
            ignore_index (Optional[int]):
                Index to ignore in the segmentation map.
            reduce_labels (bool):
                Whether to reduce the number of labels in the output.

        Returns:
            Binary masks corresponding to the segmentation map.
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
        Callable interface for preprocessing images and segmentation maps.

        Args:
            images:
                Images to preprocess.
            segmentation_maps:
                Segmentation maps associated with the images.
            **kwargs:
                Additional keyword arguments for preprocessing.

        Returns:
            Preprocessed batch of features.
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
        """
        Internal preprocessing function for handling various transformations on images.

        Args:
            image (ImageInput):
                Input image to preprocess.
            do_resize (bool, optional):
                Whether to resize the image.
            size (Dict[str, int], optional):
                Desired size for resizing (width, height).
            size_divisor (int, optional):
                Divisor for resizing the image dimensions.
            resample (PILImageResampling, optional):
                Resampling method for resizing.
            do_rescale (bool, optional):
                Whether to rescale the image.
            rescale_factor (float, optional):
                Scaling factor for image rescaling.
            do_normalize (bool, optional):
                Whether to normalize the image.
            image_mean (Union[float, List[float]], optional):
                Mean values for image normalization.
            image_std (Union[float, List[float]], optional):
                Standard deviation values for image normalization.
            input_data_format (Union[str, ChannelDimension], optional):
                Format of the input image data.

        Returns:
            Preprocessed image based on the specified transformations.
        """
    ):
        # 如果需要调整大小，则调用 resize 方法对图像进行调整
        if do_resize:
            image = self.resize(
                image, size=size, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format
            )
        # 如果需要重新缩放，则调用 rescale 方法对图像进行重新缩放
        if do_rescale:
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        # 如果需要归一化，则调用 normalize 方法对图像进行归一化
        if do_normalize:
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # 返回预处理后的图像
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
        # 将图像转换为 numpy 数组，因为所有的转换操作都要求输入为 numpy 数组
        image = to_numpy_array(image)
        # 如果图像已经进行了缩放，并且需要进行重新缩放，则记录警告信息
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 推断图像的通道格式（数据格式）如果未指定
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
        # 如果指定了数据格式，则将图像转换为该格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回预处理后的图像 numpy 数组
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
        # 如果分割地图的维度为2，添加通道维度，因为某些变换需要
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果输入数据格式未指定，根据分割地图推断通道维度的格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # TODO: (Amy)
        # 重新设计分割地图处理过程，包括减少标签数量和大小调整，不丢弃大于255的分割ID。
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
        # 如果为了处理而添加了额外的通道维度，则去除它
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
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
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        # 以下代码被复制自 transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个方法，用于将图像用零填充到指定的大小
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
        # 定义填充方式为在顶部和左侧不填充，在底部填充pad_bottom行，在右侧填充pad_right列
        padding = ((0, pad_bottom), (0, pad_right))
        # 使用指定的填充方式对图像进行填充
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

    # 以下代码段是从transformers.models.vilt.image_processing_vilt.ViltImageProcessor.pad中复制而来
    # 定义一个函数pad，用于图像的填充处理
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
        # Calculate the maximum height and width required for padding
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
        
        # Prepare data dictionary to store padded images
        data = {"pixel_values": padded_images}

        # Optionally, generate pixel masks for the padded images
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return BatchFeature object containing padded images and masks (if generated)
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
        Encodes input data into a format suitable for model input, optionally handling segmentation maps and instance IDs.

        Args:
            pixel_values_list (`List[ImageInput]`):
                List of images to encode.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation maps corresponding to images.
            instance_id_to_semantic_id (`Optional[Union[List[Dict[int, int]], Dict[int, int]]]`, *optional*):
                Mapping from instance IDs to semantic IDs.
            ignore_index (`Optional[int]`, *optional*):
                Index to ignore during encoding.
            reduce_labels (`bool`, *optional*, defaults to `False`):
                Whether to reduce the number of unique labels.
            return_tensors (`Optional[Union[str, TensorType]]`, *optional*):
                The type of tensors to return (e.g., `'tf'`, `'pt'`, `'np'`, `'jax'`).
            input_data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The channel dimension format of the input data.

        Returns:
            BatchFeature:
                Encoded inputs wrapped in a `BatchFeature` object.
        """
        # Function implementation is omitted for brevity in the comment block

    def post_process_segmentation(
        self, outputs: "MaskFormerForInstanceSegmentationOutput", target_size: Tuple[int, int] = None
    ):
        """
        Post-processes segmentation outputs to adjust them to a target size if specified.

        Args:
            outputs (`MaskFormerForInstanceSegmentationOutput`):
                Model outputs to post-process.
            target_size (`Tuple[int, int]`, *optional*):
                Target size to resize the outputs.

        """
        # Function implementation is omitted for brevity in the comment block
        ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image segmentation predictions. Only
        supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].

            target_size (`Tuple[int, int]`, *optional*):
                If set, the `masks_queries_logits` will be resized to `target_size`.

        Returns:
            `torch.Tensor`:
                A tensor of shape (`batch_size, num_class_labels, height, width`).
        """
        # Emit a warning about deprecation of this function
        logger.warning(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`",
            FutureWarning,
        )

        # class_queries_logits has shape [BATCH, QUERIES, CLASSES + 1]
        class_queries_logits = outputs.class_queries_logits
        # masks_queries_logits has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_queries_logits = outputs.masks_queries_logits

        # Resize masks if target_size is provided
        if target_size is not None:
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Remove the null class from class_queries_logits
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]

        # Calculate mask probabilities
        masks_probs = masks_queries_logits.sigmoid()

        # Perform segmentation by combining class probabilities and mask probabilities
        # using Einstein summation notation
        # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
        # where $ softmax(p) \in R^{q, c} $ is the mask classes
        # and $ sigmoid(m) \in R^{q, h, w}$ is the mask probabilities
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        return segmentation
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
    ) -> "torch.Tensor":
        """
        Post-processes outputs of an instance segmentation model, optionally converting them into semantic segmentation maps.
        
        Args:
            outputs ([MaskFormerForInstanceSegmentation]):
                Raw outputs from the instance segmentation model.
            threshold (float):
                Threshold value for class probability to consider predictions.
            mask_threshold (float):
                Threshold value for mask probabilities to consider the mask prediction.
            overlap_mask_area_threshold (float):
                Threshold for overlapping mask areas.
            target_sizes (List[Tuple[int, int]], optional):
                List specifying the desired output sizes (height, width) for each prediction.
                If `None`, predictions will not be resized.
            return_coco_annotation (bool, optional):
                Flag indicating whether to return COCO-style annotations.
            return_binary_maps (bool, optional):
                Flag indicating whether to return binary maps along with semantic segmentation.

        Returns:
            List[torch.Tensor]:
                List of semantic segmentation maps, each of shape (height, width), corresponding to the target_sizes
                entries if specified. Each entry contains semantic class IDs.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Compute segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                # Resize logits using bilinear interpolation
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # Obtain semantic segmentation map by selecting class with highest probability
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # If target_sizes is None, directly compute semantic segmentation maps
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
    # 定义一个方法用于后处理全景分割的输出结果
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
```