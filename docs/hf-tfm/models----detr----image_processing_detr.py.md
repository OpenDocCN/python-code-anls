# `.\models\detr\image_processing_detr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 以“原样”分发，无任何明示或暗示的保证或条件。
# 有关许可证的特定语言，详见许可证。
"""DETR 的图像处理器类。"""

# 导入所需模块和库
import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 导入其他模块
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    center_to_corners_format,
    corners_to_center_format,
    id_to_rgb,
    pad,
    rescale,
    resize,
    rgb_to_id,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_annotations,
)
from ...utils import (
    TensorType,
    is_flax_available,
    is_jax_tensor,
    is_scipy_available,
    is_tf_available,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_vision_available,
    logging,
)

# 如果可用，导入 Torch 库
if is_torch_available():
    import torch
    from torch import nn

# 如果可用，导入 PIL 库
if is_vision_available():
    import PIL

# 如果可用，导入 scipy 库
if is_scipy_available():
    import scipy.special
    import scipy.stats

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 支持的注释格式
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)

# 根据原始仓库：https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/datasets/transforms.py#L76
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    计算输出图像尺寸，给定输入图像尺寸和期望的输出尺寸。

    Args:
        image_size (`Tuple[int, int]`):
            输入图像尺寸。
        size (`int`):
            期望的输出尺寸。
        max_size (`int`, *optional*):
            允许的最大输出尺寸。
    """
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
```  
    # 如果高度小于等于宽度且高度等于指定大小，或者宽度小于等于高度且宽度等于指定大小，则返回高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    # 如果宽度小于高度，则按比例调整宽度和高度，保持宽度为指定大小，高度等比例缩放
    if width < height:
        ow = size
        oh = int(size * height / width)
    else:  # 如果高度小于等于宽度，则按比例调整高度和宽度，保持高度为指定大小，宽度等比例缩放
        oh = size
        ow = int(size * width / height)
    # 返回调整后的高度和宽度
    return (oh, ow)
def get_resize_output_image_size(
    input_image: np.ndarray,  # 输入的图像数组
    size: Union[int, Tuple[int, int], List[int]],  # 期望的输出图像尺寸
    max_size: Optional[int] = None,  # 允许的最大输出尺寸，可选参数
    input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道维度格式，可选参数
) -> Tuple[int, int]:  # 返回一个元组，包含宽和高
    """
    根据输入图像尺寸和期望的输出尺寸计算输出图像尺寸。如果期望的输出尺寸是元组或列表，则直接返回。如果期望的输出尺寸是整数，则通过保持输入图像尺寸的纵横比来计算输出图像尺寸。

    Args:
        input_image (`np.ndarray`):  # 输入的图像数组
            要调整尺寸的图像。
        size (`int` or `Tuple[int, int]` or `List[int]`):  # 期望的输出尺寸
            期望的输出尺寸。
        max_size (`int`, *optional*):  # 允许的最大输出尺寸，可选参数
            允许的最大输出尺寸。
        input_data_format (`ChannelDimension` or `str`, *optional*):  # 输入图像的通道维度格式，可选参数
            输入图像的通道维度格式。如果未提供，则将从输入图像中推断出来。
    """
    image_size = get_image_size(input_image, input_data_format)  # 获取输入图像的尺寸
    if isinstance(size, (list, tuple)):  # 如果期望的输出尺寸是列表或元组，则直接返回
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)  # 根据纵横比计算输出尺寸


def get_numpy_to_framework_fn(arr) -> Callable:
    """
    返回一个将 numpy 数组转换为输入数组框架的函数。

    Args:
        arr (`np.ndarray`):  # 要转换的数组
            要转换的数组。
    """
    if isinstance(arr, np.ndarray):  # 如果是 numpy 数组，则返回 np.array
        return np.array
    if is_tf_available() and is_tf_tensor(arr):  # 如果可用 TensorFlow 并且是 TensorFlow 张量，则返回 tf.convert_to_tensor
        import tensorflow as tf

        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):  # 如果可用 PyTorch 并且是 PyTorch 张量，则返回 torch.tensor
        import torch

        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):  # 如果可用 Flax 并且是 JAX 张量，则返回 jnp.array
        import jax.numpy as jnp

        return jnp.array
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")  # 抛出值错误，表示无法转换此类型的数组


def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    挤压数组，但仅当指定的轴维度为 1 时。
    """
    if axis is None:  # 如果未指定轴，则使用默认的 arr.squeeze() 方法
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)  # 尝试按指定的轴挤压数组
    except ValueError:
        return arr  # 如果指定的轴维度不为 1，则直接返回数组


def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size  # 获取图像的高度和宽度
    norm_annotation = {}  # 创建空的标准化注释字典
    for key, value in annotation.items():  # 遍历注释字典的键值对
        if key == "boxes":  # 如果键是 "boxes"
            boxes = value  # 获取边界框数组
            boxes = corners_to_center_format(boxes)  # 将边界框转换为中心-宽度格式
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)  # 将边界框归一化
            norm_annotation[key] = boxes  # 更新字典中的 "boxes" 键对应的值
        else:  # 如果键不是 "boxes"，则直接更新字典中的值
            norm_annotation[key] = value
    return norm_annotation  # 返回标准化后的注释字典


# 从 transformers.models.vilt.image_processing_vilt.max_across_indices 复制
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值的所有索引的最大值。
    """
    return [max(values_i) for values_i in zip(*values)]  # 返回每个索引处的最大值组成的列表
# 从transformers.models.vilt.image_processing_vilt.get_max_height_width复制而来
# 获取批量图片中的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批量图片中的最大高度和宽度。
    """
    # 如果未指定输入数据格式，则使用推断的通道维格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据不同的通道维格式，计算最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# 从transformers.models.vilt.image_processing_vilt.make_pixel_mask复制而来
# 创建像素遮罩
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    为图像创建像素遮罩，其中1表示有效像素，0表示填充。

    Args:
        image (`np.ndarray`):
            要创建像素遮罩的图像。
        output_size (`Tuple[int, int]`):
            遮罩的输出大小。
    """
    # 获取图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个与输出大小相同的全零遮罩
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效像素位置设为1
    mask[:input_height, :input_width] = 1
    return mask


# 受https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L33启发
# 将COCO多边形标注转换为遮罩
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    将COCO多边形标注转换为遮罩。

    Args:
        segmentations (`List[List[float]]`):
            多边形的列表，每个多边形由x-y坐标列表表示。
        height (`int`):
            遮罩的高度。
        width (`int`):
            遮罩的宽度。
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    for polygons in segmentations:
        # 根据多边形和高度宽度创建RLE编码
        rles = coco_mask.frPyObjects(polygons, height, width)
        # 解码RLE编码，创建遮罩
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        # 转换为二维遮罩
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks


# 受https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50启发
# 准备COCO检测注释，以符合DETR所期望的格式
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    将COCO格式中的目标转换为DETR所期望的格式。
    """
    # 获取图像的高度和宽度，根据输入数据格式的通道维度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取图像的ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注释
    annotations = target["annotations"]
    # 过滤掉"iscrowd"属性存在且值为0的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取注释中所有对象的类别标签
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 为了转换为COCO API，获取每个对象的面积和"iscrowd"属性
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取每个对象的边界框
    boxes = [obj["bbox"] for obj in annotations]
    # 防止没有边界框导致的大小调整
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 将边界框坐标从[x_min, y_min, width, height]格式转换为[x_min, y_min, x_max, y_max]
    boxes[:, 2:] += boxes[:, :2]
    # 将边界框坐标限制在图像范围内
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 保留有效边界框（高度和宽度大于0）
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果注释存在且第一个对象具有关键点信息
    if annotations and "keypoints" in annotations[0]:
        # 获取每个对象的关键点
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将过滤后的关键点列表转换为numpy数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 使用保留的掩码过滤相关注释
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        # 如果有关键点，则重塑为(-1, 3)的形状
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码
    if return_segmentation_masks:
        # 获取每个对象的分割掩码
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        # 将COCO多边形分割转换为掩码
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    # 返回新的目标字典
    return new_target
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    # 如果没有提供任何掩码，则返回一个空的数组表示没有边界框
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取掩码的高度和宽度
    h, w = masks.shape[-2:]
    # 创建一个行向量表示高度
    y = np.arange(0, h, dtype=np.float32)
    # 创建一个列向量表示宽度
    x = np.arange(0, w, dtype=np.float32)
    # 创建高度和宽度的网格矩阵，使用 "ij" 索引来匹配 PyTorch 中的索引顺序
    y, x = np.meshgrid(y, x, indexing="ij")

    # 按元素乘以掩码的 X 坐标
    x_mask = masks * np.expand_dims(x, axis=0)
    # 按行将结果展平，并计算每行的最大值，这是 X 坐标的最大值
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    # 创建掩码的掩码数组，然后使用掩码来填充未定义的值
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    # 按行将结果展平，并计算每行的最小值，这是 X 坐标的最小值
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 类似地，为 Y 坐标执行与 X 坐标相同的操作
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 返回堆叠的最小和最大值，形成边界框数组
    return np.stack([x_min, y_min, x_max, y_max], 1)


def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for DETR.
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 获取掩码的路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    # 创建新的目标字典
    new_target = {}
    # 将图像 ID 存储为 int64 数组
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 将图像大小存储为 int64 数组
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 将原始图像大小存储为 int64 数组
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    # 如果目标字典中包含 "segments_info" 键
    if "segments_info" in target:
        # 从文件中加载掩码并转换为数组
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        # 获取每个分段信息中的 ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 通过比较掩码和 ID，将掩码转换为二进制数组
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        # 如果需要返回掩码，则将掩码存储在目标字典中
        if return_masks:
            new_target["masks"] = masks
        # 计算边界框，并存储在目标字典中
        new_target["boxes"] = masks_to_boxes(masks)
        # 存储类别标签
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 存储 iscrowd 信息
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 存储区域信息
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    # 返回新的目标字典
    return new_target


def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入尺寸
    h, w = input_size
    # 获取目标尺寸
    final_h, final_w = target_size

    # 对 masks 进行转置，然后进行 softmax 计算
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果没有检测到任何掩模，则将 m_id 初始化为全零数组
    if m_id.shape[-1] == 0:
        # We didn't detect any mask :(
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取最大概率的索引值，并将结果 reshape 成输入尺寸的形状
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重
    if deduplicate:
        # 合并相同类别对应的掩模
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将索引映射成 RGB 图像
    seg_img = id_to_rgb(m_id)
    # 调整图像尺寸为最终尺寸
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    # 返回分割图像
    return seg_img
# 根据分割图像、目标尺寸和类别数计算掩码区域
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 获取目标尺寸
    final_h, final_w = target_size
    # 将分割图像转换为无符号8位整型
    np_seg_img = seg_img.astype(np.uint8)
    # 重塑分割图像的形状
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 使用 rgb_to_id 函数将图像转换成 id
    m_id = rgb_to_id(np_seg_img)
    # 计算每个类别的区域
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area


# 根据类别概率得分计算标签
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 使用 softmax 函数计算概率
    probs = scipy.special.softmax(logits, axis=-1)
    # 获取具有最高概率的标签
    labels = probs.argmax(-1, keepdims=True)
    # 根据标签获取得分
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 去除多余的维度
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# 对每个样本的输出进行后处理，得到视觉示例和类别预测
def post_process_panoptic_sample(
    out_logits: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    processed_size: Tuple[int, int],
    target_size: Tuple[int, int],
    is_thing_map: Dict,
    threshold=0.85,
) -> Dict:
    """
    将 [`DetrForSegmentation`] 的输出转换为单个样本的全景分割预测。

    Args:
        out_logits (`torch.Tensor`):
            该样本的逻辑。
        masks (`torch.Tensor`):
            该样本的预测分割掩码。
        boxes (`torch.Tensor`):
            该样本的预测边界框。边界框以归一化格式 `(中心_x, 中心_y, 宽度, 高度)` 给出，取值范围为 `[0, 1]`，相对于图像的大小（忽略填充）。
        processed_size (`Tuple[int, int]`):
            图像的处理大小 `(高度, 宽度)`，由预处理步骤返回，即数据增强后但未进行批处理之前的大小。
        target_size (`Tuple[int, int]`):
            图像的目标大小，`(高度, 宽度)` 对应于预测的最终大小。
        is_thing_map (`Dict`):
            一个将类别索引映射到一个布尔值的字典，指示类别是否为实物。
        threshold (`float`, *可选*，默认为 0.85):
            用于对分割掩码进行二值化的阈值。
    """
    # 筛选出空查询和低于阈值的检测
    scores, labels = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # 可能存在同一物体类别的多个预测掩码
    # 在下面，我们跟踪每个物体类别的掩码 id 列表（之后会合并）
    cur_masks = cur_masks.reshape(b, -1)
    # 使用 defaultdict 创建一个空的键值对应列表的字典
    stuff_equiv_classes = defaultdict(list)
    # 对当前类别列表进行遍历
    for k, label in enumerate(cur_classes):
        # 如果不是物品类别，则将索引添加到对应的 stuff_equiv_classes
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    # 获取分割图像
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    # 计算当前掩模的面积
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # 过滤掉太小的掩模
    if cur_classes.size() > 0:
        # 创建布尔数组用于标记面积是否小于等于4
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        # 当有小于等于4的面积时执行循环
        while filtered_small.any():
            # 根据过滤后的小面积进行筛选
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果 cur_classes 为空，则将其设置为包含一个元素的数组，数据类型为 int64
        cur_classes = np.ones((1, 1), dtype=np.int64)

    # 构建 segments_info 列表，包含 id、isthing、category_id 和 area
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除 cur_classes 变量
    del cur_classes

    # 使用 BytesIO 创建一个文件对象进行保存
    with io.BytesIO() as out:
        # 将分割图像数组转换成 PIL.Image 格式，并保存为 PNG 格式到文件对象 out
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 构建包含 PNG 字符串和 segments_info 的预测结果字典
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    # 返回预测结果
    return predictions
# 调整注释大小
def resize_annotation(
    annotation: Dict[str, Any],  # 输入参数：标注字典
    orig_size: Tuple[int, int],  # 输入参数：原始图像大小
    target_size: Tuple[int, int],  # 输入参数：目标图像大小
    threshold: float = 0.5,  # 输入参数：阈值，默认为0.5
    resample: PILImageResampling = PILImageResampling.NEAREST,  # 输入参数：重新采样滤波器，默认为最近邻插值
):
    """
    调整标注大小

    Args:
        annotation (`Dict[str, Any]`):   # 标注字典
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):    # 原始图像大小
            The original size of the input image.
        target_size (`Tuple[int, int]`):   # 目标图像大小
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):   # 阈值
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):   # 重新采样滤波器
            The resampling filter to use when resizing the masks.
    """

    # 计算尺寸比率
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    new_annotation["size"] = target_size

    # 遍历标注字典的每个键值对
    for key, value in annotation.items():
        if key == "boxes":  # 如果键是“boxes”
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)  # 缩放边界框
            new_annotation["boxes"] = scaled_boxes
        elif key == "area":  # 如果键是“area”
            area = value
            scaled_area = area * (ratio_width * ratio_height)  # 缩放区域
            new_annotation["area"] = scaled_area
        elif key == "masks":  # 如果键是“masks”
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])  # 调整大小并重新采样mask
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            new_annotation["masks"] = masks
        elif key == "size":  # 如果键是“size”
            new_annotation["size"] = target_size
        else:  # 其他情况
            new_annotation[key] = value

    return new_annotation  # 返回新的标注字典


# TODO - (Amy) make compatible with other frameworks
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
    if is_torch_tensor(mask):  # 如果输入是torch张量
        mask = mask.numpy()  # 将其转换为numpy数组

    pixels = mask.flatten()  # 展平mask
    pixels = np.concatenate([[0], pixels, [0]])  # 在mask两端添加0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]  # 计算RLE编码
    return list(runs)  # 返回RLE编码列表


# TODO - (Amy) make compatible with other frameworks
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.
    """
    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    # 获取唯一的分割标识符
    segment_ids = torch.unique(segmentation)

    # 初始化存储运行长度编码的列表
    run_length_encodings = []
    
    # 对每个分割标识符进行遍历
    for idx in segment_ids:
        # 创建掩膜，标识出当前分割标识符对应的区域
        mask = torch.where(segmentation == idx, 1, 0)
        
        # 将掩膜转换为运行长度编码
        rle = binary_mask_to_rle(mask)
        
        # 添加运行长度编码到列表中
        run_length_encodings.append(rle)

    # 返回所有分割标识符的运行长度编码列表
    return run_length_encodings
# 移除得分低于`object_mask_threshold`的对象，返回过滤后的`masks`、`scores`和`labels`
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    # 检查输入张量的第一个维度是否匹配
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")
    
    # 根据条件过滤保留的对象
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)
    
    return masks[to_keep], scores[to_keep], labels[to_keep]


def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第`k`类相关联的掩模
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询`k`中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除孤立的微小段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


def compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold: float = 0.5, overlap_mask_area_threshold: float = 0.8,
                     label_ids_to_fuse: Optional[Set[int]] = None, target_size: Tuple[int, int] = None):
    # 获取图像高度和宽度
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 创建全零的分割结果张量和空列表用于存储分段
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    # 若指定了目标大小，则插值调整掩模大小
    if target_size is not None:
        mask_probs = nn.functional.interpolate(mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)[0]

    current_segment_id = 0

    # 按照预测得分加权每个掩模
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    stuff_memory_list: Dict[str, int] = {}
    # 遍历预测标签的行数
    for k in range(pred_labels.shape[0]):
        # 获取当前预测标签的类别
        pred_class = pred_labels[k].item()
        # 检查当前预测类别是否需要进行融合
        should_fuse = pred_class in label_ids_to_fuse

        # 检查是否存在并且具有足够大小以成为一个段的蒙版
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        # 如果蒙版存在
        if mask_exists:
            # 如果当前预测类别在 stuff_memory_list 中
            if pred_class in stuff_memory_list:
                # 使用 stuff_memory_list 中的当前段 ID
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 否则增加当前段 ID
                current_segment_id += 1

            # 将当前对象段添加到最终分割地图中
            segmentation[mask_k] = current_segment_id
            # 计算段的得分
            segment_score = round(pred_scores[k].item(), 6)
            # 添加段信息到 segments 列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            # 如果应该融合
            if should_fuse:
                # 将当前段 ID 存储到 stuff_memory_list 中
                stuff_memory_list[pred_class] = current_segment_id

    # 返回分割地图和段列表
    return segmentation, segments
class DetrImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            # 标注数据格式，可以是“coco_detection”或“coco_panoptic”
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            # 控制是否将图像的`(height, width)`大小调整为指定的`size`。可以在`preprocess`方法的`do_resize`参数中进行覆盖。
            Controls whether to resize the image's `(height, width)` dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            # 调整大小后图像的`(height, width)`尺寸。可以在`preprocess`方法的`size`参数中进行覆盖。
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            # 如果调整图像大小，要使用的重采样滤波器。
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            # 控制是否按指定的比例`rescale_factor`对图像进行重新缩放。可以在`preprocess`方法的`do_rescale`参数中进行覆盖。
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            # 在重新缩放图像时使用的缩放因子。
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            # 控制是否对图像进行标准化。可以在`preprocess`方法的`do_normalize`参数中进行覆盖。
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            # 在标准化图像时使用的平均值。
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            # 在标准化图像时使用的标准差。
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            # 控制是否对图像进行填充以匹配批处理中的最大图像并创建像素掩码。
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values", "pixel_mask"]
    # 初始化函数，初始化图像处理器对象
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,  # 设置格式参数，默认为COCO_DETECTION格式
        do_resize: bool = True,  # 是否进行图像大小调整，默认为True
        size: Dict[str, int] = None,  # 图像大小参数，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        do_rescale: bool = True,  # 是否进行图像重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像标准化，默认为True
        image_mean: Union[float, List[float]] = None,  # 图像均值，默认为None
        image_std: Union[float, List[float]] = None,  # 图像标准差，默认为None
        do_pad: bool = True,  # 是否进行图像填充，默认为True
        **kwargs,  # 其他参数，以字典形式传入
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:  # 如果kwargs中包含"pad_and_return_pixel_mask"
            do_pad = kwargs.pop("pad_and_return_pixel_mask")  # 从kwargs中取出"pad_and_return_pixel_mask"并赋值给do_pad
    
        if "max_size" in kwargs:  # 如果kwargs中包含"max_size"
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",  # 发出警告信息，提醒用户"max_size"参数即将被移除
            )
            max_size = kwargs.pop("max_size")  # 从kwargs中取出"max_size"并赋值给max_size
        else:
            max_size = None if size is None else 1333  # 如果size为None，则max_size为None，否则为1333
    
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}  # 如果size不为None，则size保持原样，否则初始化为默认值
        size = get_size_dict(size, max_size=max_size, default_to_square=False)  # 调用函数get_size_dict获取处理后的图像大小字典
    
        super().__init__(**kwargs)  # 调用父类的初始化函数
        self.format = format  # 设置格式属性
        self.do_resize = do_resize  # 设置是否调整大小属性
        self.size = size  # 设置图像大小属性
        self.resample = resample  # 设置图像重采样方法属性
        self.do_rescale = do_rescale  # 设置是否重新缩放属性
        self.rescale_factor = rescale_factor  # 设置缩放因子属性
        self.do_normalize = do_normalize  # 设置是否标准化属性
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN  # 设置图像均值属性
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD  # 设置图像标准差属性
        self.do_pad = do_pad  # 设置是否填充属性
    
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):  # 从字典创建图像处理器对象的类方法
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `DetrImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()  # 复制图像处理器字典
        if "max_size" in kwargs:  # 如果kwargs中包含"max_size"
            image_processor_dict["max_size"] = kwargs.pop("max_size")  # 从kwargs中取出"max_size"并更新到图像处理器字典
        if "pad_and_return_pixel_mask
    # 准备一个可以输入到DETR模型的注释
    def prepare_annotation(self, image, target, return_segmentation_masks=None, masks_path=None, format=None) -> Dict:
        """
        准备一个可以输入到DETR模型的注释
        """
        # 如果没有指定格式，则使用默认格式
        format = format if format is not None else self.format

        # 如果格式为COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果没有指定返回分割蒙版，则默认为False
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 调用准备COCO检测注释的方法
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果格式为COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果没有指定返回分割蒙版，则默认为True
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 调用准备COCO全景注释的方法
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        # 如果格式不支持，则引发错误
        else:
            raise ValueError(f"Format {format} is not supported.")
        # 返回目标注释
        return target

    # 准备方法（已废弃）
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        # 发出警告，表示该方法即将被废弃
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用准备注释的方法，不再返回图片
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return image, target

    # 将COCO多边形转换为蒙版（已废弃）
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        # 发出警告，表示该方法即将被废弃
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 准备COCO检测注释（已废弃）
    def prepare_coco_detection(self, *args, **kwargs):
        # 发出警告，表示该方法即将被废弃
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 准备COCO全景注释（已废弃）
    def prepare_coco_panoptic(self, *args, **kwargs):
        # 发出警告，表示该方法即将被废弃
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 调整大小的方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the size to resize to. Can contain the keys `shortest_edge` and `longest_edge` or
                `height` and `width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Check if 'max_size' is in kwargs, issue a warning if found
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # Get the size dictionary with the appropriate size values
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        # Determine the size based on the keys in the size dictionary
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # Raise an error if the size dictionary does not contain the required keys
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # Resize the image with the specified parameters
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image

    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        # Resize the annotation to match the resized image
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # TODO (Amy) - update to use `rescale_factor` instead of `scale`
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def rescale_image(image: np.ndarray, rescale_factor: float, data_format: Optional[ChannelDimension] = None, input_data_format: Optional[Union[str, ChannelDimension]] = None) -> np.ndarray:
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

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format.
        """
        return normalize_annotation(annotation, image_size=image_size)

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
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

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
        # 获取批量图像中最大高度和宽度，用于填充
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对每个图像进行填充操作
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
        # 构建数据字典，包含填充后的像素值
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 为每个图像生成像素掩码
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 将像素掩码添加到数据字典中
            data["pixel_mask"] = masks

        # 返回填充后的批量特征对象
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 预处理方法，用于对输入数据进行预处理操作
    def preprocess(
        self,
        # 输入的图像数据
        images: ImageInput,
        # 输入的标注数据，可以是单个标注或标注列表，默认为 None
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        # 是否返回分割掩模，默认为 None
        return_segmentation_masks: bool = None,
        # 分割掩模路径，默认为 None
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        # 是否进行调整大小，默认为 None
        do_resize: Optional[bool] = None,
        # 调整大小的尺寸，默认为 None
        size: Optional[Dict[str, int]] = None,
        # 重采样方法，默认为 None
        resample=None,  # PILImageResampling
        # 是否进行重新缩放，默认为 None
        do_rescale: Optional[bool] = None,
        # 重新缩放因子，默认为 None
        rescale_factor: Optional[Union[int, float]] = None,
        # 是否进行归一化，默认为 None
        do_normalize: Optional[bool] = None,
        # 图像均值，默认为 None
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差，默认为 None
        image_std: Optional[Union[float, List[float]] = None,
        # 是否进行填充，默认为 None
        do_pad: Optional[bool] = None,
        # 格式，默认为 None
        format: Optional[Union[str, AnnotationFormat]] = None,
        # 是否返回张量，默认为 None
        return_tensors: Optional[Union[TensorType, str]] = None,
        # 数据格式，默认为 ChannelDimension.FIRST
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        # 输入数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
    # 后处理方法 - TODO: 添加对其他框架的支持
    # 受 https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258 启发
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 发出警告，提示`post_process`方法即将被弃用，并在 Transformers v5 中移除，建议使用`post_process_object_detection`方法代替，使用`threshold=0.`可以获得相同的结果
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 获取模型输出的logits和预测框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出logits的数量与目标尺寸的数量是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标尺寸的形状是否为(batch_size, 2)
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对logits进行softmax操作，得到概率值和对应的标签
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将预测框转换为[x0, y0, x1, y1]格式
        boxes = center_to_corners_format(out_bbox)
        # 将相对坐标[0, 1]转换为绝对坐标[0, height]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 将得分、标签和框组成字典，存入列表中
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results
    # 对模型输出进行后处理，将其转换为图像分割预测结果。仅支持 PyTorch。
    def post_process_segmentation(self, outputs, target_sizes, threshold=0.9, mask_threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into image segmentation predictions. Only supports PyTorch.

        Args:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction.
            threshold (`float`, *optional*, defaults to 0.9):
                Threshold to use to filter out queries.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, and masks for an image
            in the batch as predicted by the model.
        """
        # 发出警告，提示`post_process_segmentation`已被弃用，并将在 Transformers 的 v5 版本中移除，请使用`post_process_semantic_segmentation`。
        logger.warning_once(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_semantic_segmentation`.",
        )
        # 获取输出的logits和预测的masks
        out_logits, raw_masks = outputs.logits, outputs.pred_masks
        # 空标签的索引为out_logits的最后一个维度减1
        empty_label = out_logits.shape[-1] - 1
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        # 遍历每个输出logits、masks和目标尺寸
        for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
            # 过滤掉空查询和低于阈值的检测
            cur_scores, cur_labels = cur_logits.softmax(-1).max(-1)
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            # 对当前masks进行插值，调整到目标尺寸
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            # 将预测的masks转换为二进制值
            cur_masks = (cur_masks.sigmoid() > mask_threshold) * 1

            # 构建预测结果字典，包含得分、标签和masks
            predictions = {"scores": cur_scores, "labels": cur_labels, "masks": cur_masks}
            preds.append(predictions)
        return preds

    # 受 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218 启发
    # 将 [`DetrForSegmentation`] 的输出转换为实际的实例分割预测。仅支持 PyTorch。
    def post_process_instance(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into actual instance segmentation predictions. Only supports
        PyTorch.

        Args:
            results (`List[Dict]`):
                Results list obtained by [`~DetrImageProcessor.post_process`], to which "masks" results will be added.
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            orig_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation).
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks for an
            image in the batch as predicted by the model.
        """
        # 发出警告，提示 `post_process_instance` 已弃用，并将在 Transformers 的 v5 版本中移除，请使用 `post_process_instance_segmentation`。
        logger.warning_once(
            "`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`.",
        )

        # 检查 orig_target_sizes 和 max_target_sizes 的长度是否相等
        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError("Make sure to pass in as many orig_target_sizes as max_target_sizes")
        
        # 获取 max_target_sizes 中的最大高度和宽度
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        
        # 获取输出中的预测掩码，并进行插值操作
        outputs_masks = outputs.pred_masks.squeeze(2)
        outputs_masks = nn.functional.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        
        # 将预测掩码转换为二进制值
        outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()

        # 遍历每个预测掩码，调整大小并添加到结果中
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = nn.functional.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results

    # 受 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L241 启发
    # 受 https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258 启发
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
    ):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 获取模型输出的logits和预测框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查是否提供了目标尺寸信息
        if target_sizes is not None:
            # 检查logits的批次维度是否与目标尺寸数量相同
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # 对logits进行softmax操作，获取得分和标签
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将预测框转换为[x0, y0, x1, y1]格式
        boxes = center_to_corners_format(out_bbox)

        # 将相对坐标[0, 1]转换为绝对坐标[0, height]
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # 根据阈值筛选得分高于阈值的结果
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple[int, int]] = None):
        """
        Converts the output of [`DetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If unset, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
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
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    # inspired by https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
    # 基于给定的输出进行后处理，用于对全景分割进行优化，参考了特定位置的代码
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 阈值，用于分割像素的阈值
        mask_threshold: float = 0.5,  # 掩码阈值，用于筛选掩码像素
        overlap_mask_area_threshold: float = 0.8,  # 重叠掩码区域的阈值，用于判断是否进行掩码融合
        label_ids_to_fuse: Optional[Set[int]] = None,  # 要融合的标签ID集合，可选
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标尺寸列表，可选
```