# `.\transformers\models\mask2former\image_processing_mask2former.py`

```py
# 设定文件编码格式为 UTF-8
# 版权声明及许可协议
# 该模块用于 Mask2Former 的图像处理器类

import math  # 导入数学模块
import warnings  # 导入警告模块
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理工具相关模块
from ...image_transforms import (  # 导入图像变换相关模块
    PaddingMode,
    get_resize_output_image_size,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (  # 导入图像工具相关模块
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
)
from ...utils import (  # 导入通用工具模块
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 PyTorch 可用
if is_torch_available():
    import torch  # 导入 PyTorch 模块
    from torch import nn  # 导入 PyTorch 的 nn 模块

# 以下函数从 DETR 模型中复制而来

# 从 transformers.models.detr.image_processing_detr.max_across_indices 复制
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值中所有索引的最大值。
    """
    return [max(values_i) for values_i in zip(*values)]


# 从 transformers.models.detr.image_processing_detr.get_max_height_width 复制
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批处理中所有图像的最大高度和宽度。
    """
    if input_data_format is None:  # 如果输入数据格式未指定
        input_data_format = infer_channel_dimension_format(images[0])  # 推断输入图像的通道维度格式

    if input_data_format == ChannelDimension.FIRST:  # 如果通道维度在前
        _, max_height, max_width = max_across_indices([img.shape for img in images])  # 获取最大高度和宽度
    elif input_data_format == ChannelDimension.LAST:  # 如果通道维度在后
        max_height, max_width, _ = max_across_indices([img.shape for img in images])  # 获取最大高度和宽度
    else:  # 如果通道维度格式无效
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")  # 抛出异常
    return (max_height, max_width)


# 从 transformers.models.detr.image_processing_detr.make_pixel_mask 复制
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    生成图像的像素掩码，其中 1 表示有效像素，0 表示填充。
    """
    """
    给定图像，生成像素掩模
    Args:
        image (`np.ndarray`):
            要生成像素掩模的图像。
        output_size (`Tuple[int, int]`):
            掩模的输出尺寸。
    """
    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个指定尺寸的全零数组作为掩模
    mask = np.zeros(output_size, dtype=np.int64)
    # 将掩模的部分区域设为1，以匹配输入图像的尺寸
    mask[:input_height, :input_width] = 1
    # 返回生成的掩模
    return mask
# 将给定的二进制掩码转换为运行长度编码（RLE）格式
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
    # 如果输入的 mask 是 torch.Tensor，则转换为 numpy 数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将 mask 展平为一维数组
    pixels = mask.flatten()
    # 在数组开头和结尾各加一个额外的 0
    pixels = np.concatenate([[0], pixels, [0]])
    # 找出连续不同值的索引
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 计算运行长度编码
    runs[1::2] -= runs[::2]
    # 返回结果列表
    return list(runs)


# 将给定分割图转换为运行长度编码（RLE）格式
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    # 获取唯一的 segment_ids
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    # 遍历每个 segment_id
    for idx in segment_ids:
        # 根据 segment_id 创建二进制 mask
        mask = torch.where(segmentation == idx, 1, 0)
        # 调用 binary_mask_to_rle 函数获取运行长度编码
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    # 返回结果列表
    return run_length_encodings


# 根据 object_mask_threshold 过滤掉低分数和无对象的 masks, scores 和 labels
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
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    # 检查输入张量的第一个维度是否相等
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    # 根据条件生成需保留的索引
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    # 返回过滤后的 masks, scores 和 labels
    return masks[to_keep], scores[to_keep], labels[to_keep]


# 从 transformers.models.detr.image_processing_detr.check_segment_validity 复制而来，待补充注释
# 检查分割掩码的有效性
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取 k 类对应的掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算 k 类的总面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 判断是否存在有效掩码
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除分离的小分段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 根据预测掩码、预测类别和预测得分计算分割结果
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    # 获取目标尺寸或从掩码获取尺寸
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 初始化分割图像和分段信息列表
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    # 如果提供目标尺寸，则对掩码进行插值
    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # 根据预测得分对每个掩码进行加权
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 记录每个类别的实例
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查掩码是否存在且足够大
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象分段添加到最终分割图像
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


# 将分割图像转换为二进制掩码
def convert_segmentation_map_to_binary_masks(
    # 定义变量 segmentation_map，类型为 numpy 数组
    segmentation_map: "np.ndarray",
    # 创建 instance_id_to_semantic_id 字典，键为整数，值为整数，可选参数，默认为 None
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    # 创建 ignore_index 整数，可选参数，默认为 None
    ignore_index: Optional[int] = None,
    # 创建 reduce_labels 布尔值，可选参数，默认为 False
    reduce_labels: bool = False,
    # 如果 reduce_labels 为真且 ignore_index 未提供，则引发值错误
    if reduce_labels and ignore_index is None:
        raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

    # 如果 reduce_labels 为真，则将分割图中值为 0 的像素设为 ignore_index，其余像素减 1
    if reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # 获取唯一的标签（基于输入的类别或实例标签）
    all_labels = np.unique(segmentation_map)

    # 如果 ignore_index 不为 None，则丢弃背景标签
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # 为每个对象实例生成一个二进制掩码
    binary_masks = [(segmentation_map == i) for i in all_labels]
    binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # 将实例标签转换为类别标签
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            # 获取实例标签对应的类别标签
            class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
            labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
    else:
        labels = all_labels

    # 返回二进制掩码（转换为浮点型）和标签（转换为整型）
    return binary_masks.astype(np.float32), labels.astype(np.int64)


# 从 transformers.models.maskformer.image_processing_maskformer.get_maskformer_resize_output_image_size 复制，并将 maskformer 更改为 mask2former
def get_mask2former_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    max_size: Optional[int] = None,
    size_divisor: int = 0,
    default_to_square: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    根据所需大小计算输出大小。

    Args:
        image (`np.ndarray`):
            输入图像。
        size (`int` 或 `Tuple[int, int]` 或 `List[int]` 或 `Tuple[int]`):
            输出图像的大小。
        max_size (`int`, *可选*):
            输出图像的最大尺寸。
        size_divisor (`int`, *可选*，默认为 0):
            如果给定了 `size_divisor`，输出图像的大小将被该数字整除。
        default_to_square (`bool`, *可选*，默认为 `True`):
            如果未提供大小是否默认为正方形。
        input_data_format (`ChannelDimension` 或 `str`, *可选*):
            输入图像的通道维度格式。如果未设置，将使用输入的推断格式。

    Returns:
        `Tuple[int, int]`: 输出大小。
    """
    # 获得调整后的输出图像大小
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )

    # 如果 size_divisor 大于 0，则将输出图像的高度和宽度向上取整至能够被 size_divisor 整除
    if size_divisor > 0:
        height, width = output_size
        height = int(math.ceil(height / size_divisor) * size_divisor)
        width = int(math.ceil(width / size_divisor) * size_divisor)
        output_size = (height, width)

    return output_size
class Mask2FormerImageProcessor(BaseImageProcessor):
    r"""
    构造一个 Mask2Former 图像处理器。该图像处理器可用于准备图像和可选目标供模型使用。

    这个图像处理器继承自 [`BaseImageProcessor`]，其中包含大部分主要方法。用户应该参考这个超类以获取有关这些方法的更多信息。
    # 定义函数参数列表，包括是否需要重新调整大小、调整的大小、大小除数、重采样方式、是否需要重新缩放、缩放因子、是否需要进行归一化、图片通道均值、图片通道标准差、背景像素标签、是否减少标签数
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. 
            If size is a sequence like `(width, height)`, output size will be matched to this. 
            If size is an int, smaller edge of the image will be matched to this number. 
            i.e, if `height > width`, then image will be rescaled to `(size * height / width, size)`.
        size_divisor (`int`, *optional*, defaults to 32):
            Some backbones need images divisible by a certain number. 
            If not passed, it defaults to the value used in Swin Transformer.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. 
            This can be one of `PIL.Image.Resampling.NEAREST`, `PIL.Image.Resampling.BOX`, 
            `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`, `PIL.Image.Resampling.BICUBIC` 
            or `PIL.Image.Resampling.LANCZOS`. 
            Only has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. 
            Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. 
            Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. 
            Defaults to the ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. 
            If provided, segmentation map pixels denoted with 0 (background) will be replaced with `ignore_index`.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. 
            Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). 
            The background label will be replaced by `ignore_index`.

    # 定义模型输入的名称列表
    model_input_names = ["pixel_values", "pixel_mask"]
    def __init__(
        self,
        do_resize: bool = True,  # 是否执行调整大小的操作，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，包含"shortest_edge"和"longest_edge"两个键，默认为None
        size_divisor: int = 32,  # 图像大小的除数，默认为32
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_rescale: bool = True,  # 是否执行重新缩放的操作，默认为True
        rescale_factor: float = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否执行归一化的操作，默认为True
        image_mean: Union[float, List[float]] = None,  # 图像均值，默认为None
        image_std: Union[float, List[float]] = None,  # 图像标准差，默认为None
        ignore_index: Optional[int] = None,  # 忽略的索引，默认为None
        reduce_labels: bool = False,  # 是否减少标签，默认为False
        **kwargs,
    ):
        # 若kwargs中含有"size_divisibility"，则发出警告并将其替换为"size_divisor"
        if "size_divisibility" in kwargs:
            warnings.warn(
                "The `size_divisibility` argument is deprecated and will be removed in v4.27. Please use "
                "`size_divisor` instead.",
                FutureWarning,
            )
            size_divisor = kwargs.pop("size_divisibility")
        # 若kwargs中含有"max_size"，则发出警告并将其替换为私有属性_max_size
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` argument is deprecated and will be removed in v4.27. Please use size['longest_edge']"
                " instead.",
                FutureWarning,
            )
            # 将max_size设置为私有属性，以便在预处理方法中将其作为默认值传递，同时仍然可以传递size作为整数
            self._max_size = kwargs.pop("max_size")
        else:
            self._max_size = 1333  # 默认最大大小为1333

        # 如果size为None，则将其设置为{"shortest_edge": 800, "longest_edge": self._max_size}
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        # 根据max_size获取大小字典
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        # 调用父类的初始化方法
        super().__init__(**kwargs)
        self.do_resize = do_resize  # 执行调整大小的操作
        self.size = size  # 图像大小字典
        self.resample = resample  # 重采样方法
        self.size_divisor = size_divisor  # 图像大小的除数
        self.do_rescale = do_rescale  # 执行重新缩放的操作
        self.rescale_factor = rescale_factor  # 重新缩放因子
        self.do_normalize = do_normalize  # 执行归一化的操作
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN  # 图像均值
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD  # 图像标准差
        self.ignore_index = ignore_index  # 忽略的索引
        self.reduce_labels = reduce_labels  # 是否减少标签

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `Mask2FormerImageProcessor.from_pretrained(checkpoint, max_size=800)`
        """
        # 复制image_processor_dict
        image_processor_dict = image_processor_dict.copy()
        # 如果kwargs中含有"max_size"，则将其更新到image_processor_dict中
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果kwargs中含有"size_divisibility"，则将其更新到image_processor_dict中
        if "size_divisibility" in kwargs:
            image_processor_dict["size_divisibility"] = kwargs.pop("size_divisibility")
        # 调用父类的from_dict方法
        return super().from_dict(image_processor_dict, **kwargs)
    # 该函数用于根据给定的图像和大小参数调整图像大小
    def resize(
        self,
        image: np.ndarray,  # 输入图像，以 NumPy 数组表示
        size: Dict[str, int],  # 目标大小，可以指定 height 和 width 或 shortest_edge 和 longest_edge
        size_divisor: int = 0,  # 如果不为 0，则输出图像大小将是该数的倍数
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 用于调整大小的重采样滤波器
        data_format=None,  # 输出图像的通道顺序格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道顺序格式
        **kwargs,
    ) -> np.ndarray:
        # 如果传入了 max_size 参数，则打印一个弃用警告并删除该参数
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 根据输入的 size 参数构造一个标准的大小字典
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 从 size 字典中提取 shortest_edge、longest_edge、height 和 width 参数
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 根据图像和输入参数计算输出图像的大小
        size = get_mask2former_resize_output_image_size(
            image=image,
            size=size,
            max_size=max_size,
            size_divisor=size_divisor,
            default_to_square=False,
            input_data_format=input_data_format,
        )
        
        # 使用计算出的大小调整图像大小并返回结果
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image
    # 从DETR模型的图像处理模块中复制而来，用于将图像按给定因子重新缩放
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
        # 返回按给定因子重新缩放后的图像
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # 从MaskFormer模型的图像处理模块中复制而来，用于将分割地图转换为二进制掩码
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        # 如果未指定，使用类属性中的值来确定是否要减少标签
        reduce_labels = reduce_labels if reduce_labels is not None else self.reduce_labels
        # 如果未指定，使用类属性中的值来确定是否要忽略索引
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        # 返回转换后的二进制掩码
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            reduce_labels=reduce_labels,
        )

    # 通过调用预处理方法将图像和分割地图转换为批处理特征
    def __call__(self, images, segmentation_maps=None, **kwargs) -> BatchFeature:
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
        """Preprocesses the image according to specified parameters."""
        # If do_resize is True, resize the image with specified parameters
        if do_resize:
            image = self.resize(
                image, size=size, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format
            )
        # If do_rescale is True, rescale the image with specified parameters
        if do_rescale:
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        # If do_normalize is True, normalize the image with specified parameters
        if do_normalize:
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # Return the preprocessed image
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
        # Convert the image to numpy array
        image = to_numpy_array(image)
        # Check if the image is already scaled and do_rescale is True, then warn the user
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # If input_data_format is not specified, infer it from the image
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # Preprocess the image using _preprocess method
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
        # If data_format is specified, convert the image to that format
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # Return the preprocessed image
        return image
    # 预处理单个掩码图像
    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = 0,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        # 将分割图像转换为 numpy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果分割图像维度为 2，添加一个通道维度以适应某些变换
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            # 输入数据格式为第一维度为通道维度
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果没有指定输入数据格式，则推断输入数据的通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map)
        # TODO: (Amy)
        # 重新处理分割图像以包括减少标签和调整大小，这不会丢弃大于 255 的分割 ID。
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
        # 如果为了处理而添加了额外的通道维度，则去除该维度
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        # 返回预处理后的分割图像
        return segmentation_map
    
    # 预处理图像和分割图像
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
        # 从 numpy 数组或 PIL 图像列表中获取图像，并对它们进行预处理
        ...
    
    # 从 numpy 数组扩展边界填充图像
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        ...
    def pad(self,  # 定义名为pad的函数，接受self参数和其他参数
        images: List[np.ndarray],  # images参数为numpy数组组成的列表
        constant_values: Union[float, Iterable[float]] = 0,  # constant_values参数为float类型或者可迭代的float类型，默认为0
        return_pixel_mask: bool = True,  # return_pixel_mask参数为布尔类型，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # return_tensors参数为可选的字符串或TensorType类型，默认为None
        data_format: Optional[ChannelDimension] = None,  # data_format参数为可选的ChannelDimension类型，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # input_data_format参数为可选的字符串或ChannelDimension类型，默认为None
        ) -> np.ndarray:  # 函数返回类型为numpy数组

        """
        Pad an image with zeros to the given size.  # 对图像进行零填充以达到给定的大小
        """
        
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)  # 获取输入图像的尺寸，其中channel_dim为输入数据格式
        output_height, output_width = output_size  # 获取输出尺寸

        pad_bottom = output_height - input_height  # 计算垂直方向的填充大小
        pad_right = output_width - input_width  # 计算水平方向的填充大小
        padding = ((0, pad_bottom), (0, pad_right))  # 构建填充元组
        padded_image = pad(
            image,  # 要填充的图像
            padding,  # 填充参数
            mode=PaddingMode.CONSTANT,  # 填充模式为常数填充
            constant_values=constant_values,  # 常数填充值
            data_format=data_format,  # 数据格式
            input_data_format=input_data_format,  # 输入数据格式
        )  # 执行填充操作并将结果赋给padded_image变量
        return padded_image  # 返回填充后的图像
    def pad_images(
        self, images: List[np.ndarray], constant_values: Union[float, Iterable[float]] = 0.0,
        return_pixel_mask: bool = True, return_tensors: Union[str, TensorType] = TensorType.TENSORFLOW,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            images (`List[np.ndarray]`):
                List of images to pad.
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
        # Calculate the maximum height and width among the images in the batch
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # Pad each image in the batch to the calculated pad size
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
        data = {"pixel_values": padded_images}

        # Optionally, generate pixel masks for the padded images
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return BatchFeature containing the padded images and pixel masks
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
        Encodes inputs for the model, such as images and segmentation maps.

        Args:
            pixel_values_list (`List[ImageInput]`):
                List of pixel values representing images.
            segmentation_maps (`ImageInput`, *optional*):
                Pixel values representing segmentation maps.
            instance_id_to_semantic_id (`List[Dict[int, int]]` or `Dict[int, int]`, *optional*):
                Mapping from instance ids to semantic ids.
            ignore_index (`int`, *optional*):
                Index to ignore in loss computation.
            reduce_labels (`bool`, *optional*):
                Whether to reduce the number of labels.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input data.
        """
        # Implementation omitted for brevity

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Post-processes the semantic segmentation outputs.

        Args:
            outputs: Model outputs.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of target image sizes.
        """
        # Implementation omitted for brevity
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
        # 获取类别查询的logits，形状为[batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits  
        # 获取掩码查询的logits，形状为[batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits  

        # 将尺寸缩放回预处理图像大小 - 对于所有模型都是(384, 384)
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # 移除空类别 `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # 语义分割logits的形状为(batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # 调整logits的大小并计算语义分割地图
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
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
    # 对实例分割结果进行后处理
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 设置阈值用于筛选实例分割结果
        mask_threshold: float = 0.5,  # 设置阈值用于筛选掩模
        overlap_mask_area_threshold: float = 0.8,  # 设置重叠掩模面积的阈值
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标尺寸的可选列表
        return_coco_annotation: Optional[bool] = False,  # 是否返回 COCO 标注的可选布尔值
        return_binary_maps: Optional[bool] = False,  # 是否返回二进制地图的可选布尔值
    
    # 对全景分割结果进行后处理
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 设置阈值用于筛选全景分割结果
        mask_threshold: float = 0.5,  # 设置阈值用于筛选掩模
        overlap_mask_area_threshold: float = 0.8,  # 设置重叠掩模面积的阈值
        label_ids_to_fuse: Optional[Set[int]] = None,  # 要融合的标签 ID 的可选集合
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标尺寸的可选列表
```