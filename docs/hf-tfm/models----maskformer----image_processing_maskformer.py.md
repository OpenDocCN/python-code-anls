# `.\transformers\models\maskformer\image_processing_maskformer.py`

```
# 设置编码格式为 UTF-8
# 版权声明以及版权信息
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律要求或书面同意的情况下，根据许可协议分发的软件
# 以"原样"分发，不提供任何形式的保证或条件，无论是明示或暗示的
# 有关特定语言的条件和保证，请参阅许可证
"""用于 MaskFormer 的图像处理器类"""

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 导入必要的工具函数和类
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

# 如果支持类型检查
if TYPE_CHECKING:
    from transformers import MaskFormerForInstanceSegmentationOutput

# 如果支持 torch 库，则进行相关导入
if is_torch_available():
    import torch
    from torch import nn

# 从 DETR 模型的图像处理模块中复制的函数：获取可迭代值中每个元素最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值中所有索引的最大值。
    """
    return [max(values_i) for values_i in zip(*values)]

# 从 DETR 模型的图像处理模块中复制的函数：获取所有图像中最大的高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批量图像中所有图像的最大高度和宽度。
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

# 从 DETR ���型的图像处理模块中复制的函数：创建像素级掩码
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    # 为图像创建像素掩码,其中1表示有效像素,0表示填充像素
    def make_pixel_mask(image, output_size):
        # 获取图像的输入高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 创建一个全0的掩码矩阵,大小为输出尺寸
        mask = np.zeros(output_size, dtype=np.int64)
        # 将有效像素区域(输入高度和宽度)的掩码值设为1
        mask[:input_height, :input_width] = 1
        # 返回创建的掩码矩阵
        return mask
# 从transformers.models.detr.image_processing_detr中复制的函数，将给定的二进制掩码(`height, width`)转换为运行长度编码(RLE)格式
def binary_mask_to_rle(mask):
    """
    将给定形状为`（height, width）`的二进制掩码转换为运行长度编码（RLE）格式。

    Args:
        mask (`torch.Tensor`或`numpy.array`):
            形状为`（height, width）`的二进制掩码张量，其中0表示背景，1表示目标段ID或类别ID。
    Returns:
        `List`: 二进制掩码的运行长度编码列表。有关RLE格式的更多信息，请参考COCO API。
    """
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# 从transformers.models.detr.image_processing_detr中复制的函数，将给定的分割映射(`height, width`)转换为运行长度编码（RLE）格式
def convert_segmentation_to_rle(segmentation):
    """
    将给定形状为`（height, width）`的分割映射转换为运行长度编码（RLE）格式。

    Args:
        segmentation (`torch.Tensor`或`numpy.array`):
            形状为`（height, width）`的分割映射，每个值表示一个段或类别ID。
    Returns:
        `List[List]`: 一个列表的列表，每个列表是段/类别ID的运行长度编码。
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


# 从transformers.models.detr.image_processing_detr中复制的函数，使用`object_mask_threshold`将给定掩码进行二值化，返回相应的`masks`，`scores`和`labels`值
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    使用`object_mask_threshold`对给定的掩码进行二值化，返回相关的`masks`，`scores`和`labels`值。

    Args:
        masks (`torch.Tensor`):
            形状为`（num_queries, height, width）`的张量。
        scores (`torch.Tensor`):
            形状为`（num_queries）`的张量。
        labels (`torch.Tensor`):
            形状为`（num_queries）`的张量。
        object_mask_threshold (`float`):
            介于0和1之间的数字，用于二值化掩码。
    Raises:
        `ValueError`: 当所有输入张量的第一个维度不匹配时引发。
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: 不包含区域<`object_mask_threshold`的`masks`，`scores`和`labels`。
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# 从transformers.models.detr.image_processing_detr中复制的函数，检查分段的有效性
# 检查分割的有效性，返回是否存在掩码和掩码对应的二进制数组
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第 k 类相关联的掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询 k 中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除断开的小片段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 从 transformers.models.detr.image_processing_detr.compute_segments 复制而来
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

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # 根据预测分数加权每个掩码
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查掩码是否存在且足够大以成为段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象段添加到最终分割图中
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
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


# TODO: (Amy) 移动到 image_transforms
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    # 定义一个可选的整数类型参数 ignore_index，默认值为 None
    ignore_index: Optional[int] = None,
    # 定义一个布尔类型参数 reduce_labels，默认值为 False
    reduce_labels: bool = False,
# 如果 reduce_labels 为 True，但未提供 ignore_index，则抛出 ValueError 异常
if reduce_labels and ignore_index is None:
    raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

# 如果 reduce_labels 为 True，将 segmentation_map 中值为 0 的元素替换为 ignore_index，其余元素减 1
if reduce_labels:
    segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

# 获取 segmentation_map 中的所有唯一标签
all_labels = np.unique(segmentation_map)

# 如果 ignore_index 不为 None，则从 all_labels 中删除 ignore_index
if ignore_index is not None:
    all_labels = all_labels[all_labels != ignore_index]

# 为每个对象实例生成二进制掩码
binary_masks = [(segmentation_map == i) for i in all_labels]
binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

# 将实例 id 转换为类别 id
if instance_id_to_semantic_id is not None:
    labels = np.zeros(all_labels.shape[0])

    for label in all_labels:
        class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
        labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
else:
    labels = all_labels

# 返回二进制掩码和标签
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
    Computes the output size given the desired size.

    Args:
        image (`np.ndarray`):
            The input image.
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            The size of the output image.
        max_size (`int`, *optional*):
            The maximum size of the output image.
        size_divisor (`int`, *optional*, defaults to 0):
            If `size_divisor` is given, the output image size will be divisible by the number.
        default_to_square (`bool`, *optional*, defaults to `True`):
            Whether to default to square if no size is provided.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `Tuple[int, int]`: The output size.
    """
    # 计算输出大小
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )

    # 如果 size_divisor 大于 0，将输出大小调整为可被 size_divisor 整除
    if size_divisor > 0:
        height, width = output_size
        height = int(math.ceil(height / size_divisor) * size_divisor)
        width = int(math.ceil(width / size_divisor) * size_divisor)
        output_size = (height, width)

    return output_size


class MaskFormerImageProcessor(BaseImageProcessor):
    r"""
    # 构造一个 MaskFormer 图像处理器。该图像处理器可用于准备图像和可选目标，以供模型使用。
    # 该图像处理器继承自 BaseImageProcessor，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    # 参数：
    # do_resize（`bool`，*可选*，默认为`True`）：是否调整输入大小到特定的`size`。
    # size（`int`，*可选*，默认为800）：将输入调整为给定大小。仅在`do_resize`设置为`True`时有效。如果size是一个类似`(width, height)`的序列，输出大小将匹配到这个。如果size是一个整数，图像的较小边将匹配到这个数字。即，如果`height > width`，则图像将重新缩放为`(size * height / width, size)`。
    # size_divisor（`int`，*可选*，默认为32）：一些骨干网络需要可被某个数字整除的图像。如果未传递，则默认为Swin Transformer中使用的值。
    # resample（`int`，*可选*，默认为`Resampling.BILINEAR`）：一个可选的重采样滤波器。可以是`PIL.Image.Resampling.NEAREST`、`PIL.Image.Resampling.BOX`、`PIL.Image.Resampling.BILINEAR`、`PIL.Image.Resampling.HAMMING`、`PIL.Image.Resampling.BICUBIC`或`PIL.Image.Resampling.LANCZOS`之一。仅在`do_resize`设置为`True`时有效。
    # do_rescale（`bool`，*可选*，默认为`True`）：是否将输入重新缩放到特定的`scale`。
    # rescale_factor（`float`，*可选*，默认为`1/255`）：按给定因子重新缩放输入。仅在`do_rescale`设置为`True`时有效。
    # do_normalize（`bool`，*可选*，默认为`True`）：是否对输入进行均值和标准差归一化。
    # image_mean（`int`，*可选*，默认为`[0.485, 0.456, 0.406]`）：每个通道的均值序列，用于归一化图像时使用。默认为ImageNet均值。
    # image_std（`int`，*可选*，默认为`[0.229, 0.224, 0.225]`）：每个通道的标准差序列，用于归一化图像时使用。默认为ImageNet标准差。
    # ignore_index（`int`，*可选*）：在分割地图中为背景像素分配的标签。如果提供，用0（背景）表示的分割地图像素将被替换为`ignore_index`。
    # do_reduce_labels（`bool`，*可选*，默认为`False`）：是否减少所有分割地图的标签值1。通常用于数据集中使用0表示背景，并且背景本身不包含在数据集的所有类中（例如ADE20k）。背景标签将被替换为`ignore_index`。
    # 定义模型输入的名称列表
    model_input_names = ["pixel_values", "pixel_mask"]

    # 初始化方法，设置各种参数
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
        do_reduce_labels: bool = False,
        **kwargs,
    ):
        # 检查是否有"size_divisibility"参数，如果有则发出警告并使用"size_divisor"代替
        if "size_divisibility" in kwargs:
            warnings.warn(
                "The `size_divisibility` argument is deprecated and will be removed in v4.27. Please use "
                "`size_divisor` instead.",
                FutureWarning,
            )
            size_divisor = kwargs.pop("size_divisibility")
        # 检查是否有"max_size"参数，如果有则发出警告并使用"size['longest_edge']"代替
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` argument is deprecated and will be removed in v4.27. Please use size['longest_edge']"
                " instead.",
                FutureWarning,
            )
            # 将max_size设置为私有属性，以便在预处理方法中传递默认值，同时仍然可以将`size`作为整数传递
            self._max_size = kwargs.pop("max_size")
        else:
            self._max_size = 1333
        # 检查是否有"reduce_labels"参数，如果有则发出警告并使用"do_reduce_labels"代替
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` argument is deprecated and will be removed in v4.27. Please use "
                "`do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        # 设置size参数的默认值
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        # 调用父类的初始化方法
        super().__init__(**kwargs)
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

    # 类方法
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        重写基类的 `from_dict` 方法，确保在使用 from_dict 和 kwargs 创建图像处理器时更新参数，例如 `MaskFormerImageProcessor.from_pretrained(checkpoint, max_size=800)`
        """
        # 复制传入的图像处理器字典，以免修改原始参数
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中包含 "max_size"，则更新图像处理器字典中的 "max_size" 参数
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果 kwargs 中包含 "size_divisibility"，则更新图像处理器字典中的 "size_divisibility" 参数
        if "size_divisibility" in kwargs:
            image_processor_dict["size_divisibility"] = kwargs.pop("size_divisibility")
        # 调用基类的 from_dict 方法，传入更新后的图像处理器字典和 kwargs
        return super().from_dict(image_processor_dict, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 0,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 定义一个函数，用于将图像调整大小到给定尺寸。尺寸可以是最小尺寸（标量）或`(高度，宽度)`元组。如果尺寸是一个整数，图像的较小边将匹配到这个数字。
    def resize(
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 0,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Union[ChannelDimension, str] = None,
        input_data_format: Union[ChannelDimension, str] = None,
    ):
        # 如果参数中包含`max_size`，发出警告并将其移除
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # 获取调整后的尺寸字典
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        # 如果尺寸字典中包含`shortest_edge`和`longest_edge`，则分别赋值给`size`和`max_size`
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        # 如果尺寸字典中包含`height`和`width`，则将其转换为元组形式
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            # 抛出数值错误，要求尺寸字典必须包含`height`和`width`键或`shortest_edge`和`longest_edge`键
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # 获取调整后的图像尺寸
        size = get_maskformer_resize_output_image_size(
            image=image,
            size=size,
            max_size=max_size,
            size_divisor=size_divisor,
            default_to_square=False,
            input_data_format=input_data_format,
        )
        # 调整图像大小
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        # 返回调整后的图像
        return image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale中复制的函数
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def rescale_image(
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
        segmentation_map: np.ndarray,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        """
        Convert a segmentation map to binary masks.

        Args:
            segmentation_map (`np.ndarray`):
                Segmentation map to convert.
            instance_id_to_semantic_id (`Dict[int, int]`, *optional*):
                Mapping from instance IDs to semantic IDs.
            ignore_index (`int`, *optional*):
                Index to ignore in the segmentation map.
            reduce_labels (`bool`):
                Whether to reduce the number of labels.

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
        Call the preprocess method with the given arguments.

        Args:
            images: Input images.
            segmentation_maps: Segmentation maps corresponding to the images.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchFeature object after preprocessing.
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
        Preprocess the input image with various options.

        Args:
            image: Input image to preprocess.
            do_resize: Whether to resize the image.
            size: Dictionary containing the target size for resizing.
            size_divisor: Divisor for resizing the image.
            resample: Resampling method for resizing.
            do_rescale: Whether to rescale the image.
            rescale_factor: Factor for rescaling the image.
            do_normalize: Whether to normalize the image.
            image_mean: Mean value for normalization.
            image_std: Standard deviation value for normalization.
            input_data_format: Format of the input image data.

        """
    # 如果需要调整大小，则调用resize方法
    if do_resize:
        image = self.resize(
            image, size=size, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format
        )
    # 如果需要重新缩放，则调用rescale方法
    if do_rescale:
        image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
    # 如果需要归一化，则调用normalize方法
    if do_normalize:
        image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
    # 返回处理后的图像
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
    # 将图像转换为numpy数组
    image = to_numpy_array(image)
    # 如果图像已经缩放且需要重新缩放，则发出警告
    if is_scaled_image(image) and do_rescale:
        logger.warning_once(
            "It looks like you are trying to rescale already rescaled images. If the input"
            " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        )
    # 推断输入数据格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    # 调用_preprocess方法处理图像
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
    # 如果指定了数据格式，则转换图像通道维度格式
    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    # 返回处理后的图像
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
        # 将分割图转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果分割图缺少通道维度，则添加通道维度，某些转换需要
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果输入数据格式未指定，则推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # TODO: (Amy)
        # 重新设计分割图处理流程，包括减少标签和调整大小，不会丢弃大于255的分割 ID
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
        # 如果为了处理而添加了额外的通道维度，则移除
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
        image_mean: Optional[Union[float, List[float]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image 复制而来
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个函数，用于将图像用零填充到指定大小
    def pad(
        self,
        images: List[np.ndarray],  # 输入图像列表
        constant_values: Union[float, Iterable[float]] = 0,  # 填充值，默认为0
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，默认为None
        data_format: Optional[ChannelDimension] = None,  # 数据格式，可选
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选
    )

    # 函数注释：用零填充图像到指定大小
    def pad_image(
        image: np.ndarray,  # 输入图像
        output_size: Tuple[int, int],  # 输出大小
        constant_values: Union[float, Iterable[float]] = 0,  # 填充值，默认为0
        data_format: Optional[ChannelDimension] = None,  # 数据格式，可选
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出图像的高度和宽度
        output_height, output_width = output_size

        # 计算需要在底部和右侧填充的像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充元组
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
        # 返回填充后的图像
        return padded_image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.pad复制而来
    # 函数注释：用零填充图像到指定大小
    def pad(
        self,
        images: List[np.ndarray],  # 输入图像列表
        constant_values: Union[float, Iterable[float]] = 0,  # 填充值，默认为0
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，默认为None
        data_format: Optional[ChannelDimension] = None,  # 数据格式，可选
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选
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
        # Calculate the size for padding based on the maximum height and width in the batch
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # Pad each image in the batch with zeros to match the pad_size
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
        # Prepare the data dictionary with padded images
        data = {"pixel_values": padded_images}

        # If return_pixel_mask is True, generate pixel masks for each image in the batch
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return BatchFeature object with the prepared data and tensor type
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
    def post_process_segmentation(
        self, outputs: "MaskFormerForInstanceSegmentationOutput", target_size: Tuple[int, int] = None
    ) -> "torch.Tensor":
        """
        将[`MaskFormerForInstanceSegmentationOutput`]的输出转换为图像分割预测。仅支持PyTorch。

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                [`MaskFormerForInstanceSegmentation`]的输出。

            target_size (`Tuple[int, int]`, *optional*):
                如果设置，`masks_queries_logits`将被调整为`target_size`。

        Returns:
            `torch.Tensor`:
                形状为(`batch_size, num_class_labels, height, width`)的张量。
        """
        logger.warning(
            "`post_process_segmentation`已弃用，并将在Transformers的v5中移除，请使用`post_process_instance_segmentation`",
            FutureWarning,
        )

        # class_queries_logits的形状为[BATCH, QUERIES, CLASSES + 1]
        class_queries_logits = outputs.class_queries_logits
        # masks_queries_logits的形状为[BATCH, QUERIES, HEIGHT, WIDTH]
        masks_queries_logits = outputs.masks_queries_logits
        if target_size is not None:
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # 移除空类 `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # mask probs的形状为[BATCH, QUERIES, HEIGHT, WIDTH]
        masks_probs = masks_queries_logits.sigmoid()
        # 现在我们想要对查询求和，
        # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
        # 其中 $ softmax(p) \in R^{q, c} $ 是掩码类别
        # 而 $ sigmoid(m) \in R^{q, h, w}$ 是掩码概率
        # b(atch)q(uery)c(lasses), b(atch)q(uery)h(eight)w(idth)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        return segmentation

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
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
        # 获取类别查询的logits，形状为[batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # 获取掩码查询的logits，形状为[batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        # 移除空类别 `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # 计算掩码概率，形状为[batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()

        # 计算语义分割logits，形状为(batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # 调整logits大小并计算语义分割地图
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                # 调整logits大小
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 获取语义地图
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # 获取语义分割结果
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
    # 对全景分割输出进行后处理
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 阈值设为0.5
        mask_threshold: float = 0.5,  # 掩码阈值设为0.5
        overlap_mask_area_threshold: float = 0.8,  # 重叠掩码面积阈值设为0.8
        label_ids_to_fuse: Optional[Set[int]] = None,  # 要融合的标签ID集合，默认为None
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标尺寸列表，默认为None
```