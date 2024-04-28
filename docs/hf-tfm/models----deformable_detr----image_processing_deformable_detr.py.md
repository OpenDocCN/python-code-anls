# `.\models\deformable_detr\image_processing_deformable_detr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 许可证信息
"""Deformable DETR 的图像处理器类"""

# 导入所需模块
import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 从相关模块导入所需函数或类
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
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

# 如果可用，导入 torch 模块和 nn 类
if is_torch_available():
    import torch
    from torch import nn

# 如果可用，导入 PIL 模块
if is_vision_available():
    import PIL

# 如果可用，导入 scipy 模块
if is_scipy_available():
    import scipy.special
    import scipy.stats

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 支持的注释格式
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# 从transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio中获取图像大小和纵横比
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    根据输入图像大小和所需的输出大小计算输出图像大小。

    参数:
        image_size (`Tuple[int, int]`):
            输入图像大小。
        size (`int`):
            所需的输出大小。
        max_size (`int`, *optional*):
            允许的最大输出大小。
    """
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    # 如果高度小于等于宽度且高度等于指定尺寸，或者宽度小于等于高度且宽度等于指定尺寸，则返回高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    
    # 如果宽度小于高度，则计算调整后的宽度和高度
    if width < height:
        # 新的调整宽度为指定尺寸
        ow = size
        # 计算调整后的高度，按比例缩放
        oh = int(size * height / width)
    # 如果宽度大于等于高度，则计算调整后的高度和宽度
    else:
        # 新的调整高度为指定尺寸
        oh = size
        # 计算调整后的宽度，按比例缩放
        ow = int(size * width / height)
    # 返回调整后的高度和宽度
    return (oh, ow)
# 从transformers.models.detr.image_processing_detr.get_resize_output_image_size复制而来
# 根据输入图像大小和期望的输出大小计算输出图像大小。如果期望的输出大小是元组或列表，则直接返回输出图像大小。如果期望的输出大小是整数，则通过保持输入图像大小的纵横比来计算输出图像大小。
def get_resize_output_image_size(
    input_image: np.ndarray,  # 输入图像
    size: Union[int, Tuple[int, int], List[int]],  # 期望的输出大小
    max_size: Optional[int] = None,  # 最大允许的输出大小
    input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道尺寸格式，如果未提供，将从输入图像中推断出来
) -> Tuple[int, int]:  # 返回值为元组(int, int)类型的输出图像大小
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or `List[int]`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """

# 从transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn复制而来
# 返回一个函数，该函数将numpy数组转换为输入数组的框架
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# 从transformers.models.detr.image_processing_detr.safe_squeeze复制而来
# 只有在指定的轴具有维度1时才压缩数组
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# 从transformers.models.detr.image_processing_detr.normalize_annotation复制而来
# 根据图像大小规范化注释
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    norm_annotation = {}
    # 遍历注释字典的键值对
    for key, value in annotation.items():
        # 如果键为"boxes"
        if key == "boxes":
            # 将值赋给变量boxes
            boxes = value
            # 使用corners_to_center_format函数将边界框坐标格式转换为中心点坐标格式
            boxes = corners_to_center_format(boxes)
            # 将边界框坐标归一化为图像宽高的比例
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的边界框值存入norm_annotation字典中
            norm_annotation[key] = boxes
        # 如果键不为"boxes"
        else:
            # 将值存入norm_annotation字典中
            norm_annotation[key] = value
    # 返回归一化后的注释字典
    return norm_annotation
# Copied from transformers.models.detr.image_processing_detr.max_across_indices
# 根据给定的可迭代值，返回所有索引上的最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
# 获取批处理中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果输入数据格式为空，则推断图像的通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 如果输入数据格式是通道维度在前，则获取最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    # 如果输入数据格式是通道维度在后，则获取最大高度和宽度
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    # 否则，引发值错误
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
# 为图像创建像素掩码，其中1表示有效像素，0表示填充像素
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
    # 获取图像的输入高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个具有指定输出大小和int64类型的零数组
    mask = np.zeros(output_size, dtype=np.int64)
    # 将1分配给输入高度和宽度内的像素
    mask[:input_height, :input_width] = 1
    return mask


# Copied from transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask
# 将COCO多边形注释转换为掩码
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon annotation to a mask.

    Args:
        segmentations (`List[List[float]]`):
            List of polygons, each polygon represented by a list of x-y coordinates.
        height (`int`):
            Height of the mask.
        width (`int`):
            Width of the mask.
    """
    try:
        # 如果导入失败，则引发导入错误
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    # 遍历多边形列表
    for polygons in segmentations:
        # 使用多边形列表、高度和宽度创建RLE编码
        rles = coco_mask.frPyObjects(polygons, height, width)
        # 解码RLE编码，返回掩码
        mask = coco_mask.decode(rles)
        # 如果掩码的维度小于3，则添加一个维度
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # 将掩码转换为uint8类型的数组
        mask = np.asarray(mask, dtype=np.uint8)
        # 按轴2的方向上的任意元素进行逻辑或运算
        mask = np.any(mask, axis=2)
        masks.append(mask)
    # 将所有掩码堆叠成一个数组
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks
# 从transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation复制而来，将DETR格式的目标转换为DeformableDetr所期望的格式
def prepare_coco_detection_annotation(
    image,  # 图像数据
    target,  # 目标数据
    return_segmentation_masks: bool = False,  # 是否返回分割掩码
    input_data_format: Optional[Union[ChannelDimension, str]] = None,  # 输入数据格式
):
    """
    将COCO格式的目标转换为DeformableDetr所期望的格式。
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取图像的ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注释
    annotations = target["annotations"]
    # 过滤掉"isocrowd"属性为0或不存在的注释
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有目标类别
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 为了转换为COCO API格式
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取目标边界框
    boxes = [obj["bbox"] for obj in annotations]
    # 防止没有边界框通过调整大小
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 保留有效边界框
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果存在关键点，则添加关键点信息
    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将过滤后的关键点列表转换为numpy数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 在此处应用保留掩码以过滤相关注释
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码，则添加分割掩码信息
    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


# 从transformers.models.detr.image_processing_detr.masks_to_boxes复制而来
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩码周围的边界框。

    Args:
        masks: 格式为`[number_masks, height, width]`的掩码，其中N是掩码数量

    Returns:
        boxes: 格式为`[number_masks, 4]`的边界框，xyxy格式
    """
```  
    # 如果 masks 的大小为 0，则返回形状为 (0, 4) 的零数组
    if masks.size == 0:
        return np.zeros((0, 4))
    
    # 获取 masks 的高度 h 和宽度 w
    h, w = masks.shape[-2:]
    # 创建包含 0 到 h-1 之间所有整数的一维数组 y
    y = np.arange(0, h, dtype=np.float32)
    # 创建包含 0 到 w-1 之间所有整数的一维数组 x
    x = np.arange(0, w, dtype=np.float32)
    # 生成网格点坐标矩阵，使用 "ij" 索引
    y, x = np.meshgrid(y, x, indexing="ij")
    
    # 计算 x 方向的最大值和最小值
    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)
    
    # 计算 y 方向的最大值和最小值
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)
    
    # 返回合并后的最小值和最大值数组
    return np.stack([x_min, y_min, x_max, y_max], 1)
# 从 transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation 复制，将 DETR 替换为 DeformableDetr
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    为 DeformableDetr 准备 coco 全景注释。
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 构建注释路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    # 将 image_id 转换为数组形式，若不存在则使用 id
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 将图像尺寸转换为数组形式
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 将原始图像尺寸转换为数组形式
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        # 读取注释图像并转换为数组形式
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将 RGB 转换为 ID
        masks = rgb_to_id(masks)

        # 获取每个分段信息的 ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 将每个分段的掩码与对应的 ID 匹配
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        # 如果需要返回掩码，则存储在新目标字典中
        if return_masks:
            new_target["masks"] = masks
        # 将掩码转换为边界框并存储在新目标字典中
        new_target["boxes"] = masks_to_boxes(masks)
        # 将类别标签存储在新目标字典中
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 将是否是群集存储在新目标字典中
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 将面积信息存储在新目标字典中
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


# 从 transformers.models.detr.image_processing_detr.get_segmentation_image 复制
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入图像的高度和宽度
    h, w = input_size
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size

    # 对掩码进行 softmax 操作
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    if m_id.shape[-1] == 0:
        # 如果没有检测到任何掩码，则创建全零掩码
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 将掩码转换为 ID，并且将其重新形状以匹配输入图像的尺寸
        m_id = m_id.argmax(-1).reshape(h, w)

    if deduplicate:
        # 合并属于相同物体类别的掩码
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将 ID 转换为 RGB 图像
    seg_img = id_to_rgb(m_id)
    # 将分割图像调整为目标尺寸
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# 从 transformers.models.detr.image_processing_detr.get_mask_area 复制
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size
    # 将分割图像转换为 numpy 数组
    np_seg_img = seg_img.astype(np.uint8)
    # 重新形状以匹配目标图像的尺寸
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 将 RGB 图像转换为 ID
    m_id = rgb_to_id(np_seg_img)
    # 计算每个类别的样本数量，m_id == i 表示将类别标签与当前索引 i 比较，得到布尔值数组，sum() 统计 True 的数量即为该类别的样本数量
    area = [(m_id == i).sum() for i in range(n_classes)]
    # 返回每个类别的样本数量列表
    return area
# 从类别概率中计算标签得分
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 使用 scipy.special.softmax 计算概率
    probs = scipy.special.softmax(logits, axis=-1)
    # 取最大概率对应的类别标签
    labels = probs.argmax(-1, keepdims=True)
    # 从概率中获取对应标签的得分
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 压缩得分和标签的维度
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    # 返回得分和标签
    return scores, labels

# 对单个样本的输出进行后处理，生成全景分割预测
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
            该样本的logits。
        masks (`torch.Tensor`):
            该样本的预测分割掩模。
        boxes (`torch.Tensor`):
            该样本的预测边界框。边界框以规范化格式 `(中心坐标_x, 中心坐标_y, 宽度, 高度)` 给出，取值范围 `[0, 1]`，相对于图像的大小（不考虑填充）。
        processed_size (`Tuple[int, int]`):
            图像的处理尺寸 `(高度, 宽度)`，由预处理步骤返回，即数据增强后但未批处理之前的大小。
        target_size (`Tuple[int, int]`):
            图像的目标尺寸，`(高度, 宽度)` 对应于请求的最终预测大小。
        is_thing_map (`Dict`):
            将类别索引映射到一个布尔值，指示类别是物体还是非物体。
        threshold (`float`, *可选参数*, 默认为 0.85):
            用于二值化分割掩模的阈值。
    """
    # 过滤空查询和得分低于阈值的检测
    scores, labels = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    cur_masks = masks[keep]
    # 对掩模进行调整大小
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # 可能会有多个相同类别的预测掩模，此处跟踪每个类别的掩模id列表（稍后将合并）
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    # 对当前类别进行枚举，获取类别索引和标签
    for k, label in enumerate(cur_classes):
        # 如果不是物体类别，则将其添加到对应的“stuff”类别中
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    # 获取分割图像
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    # 获取每个分割区域的面积
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # 过滤面积过小的分割区域
    if cur_classes.size() > 0:
        # 过滤掉面积小于等于4的分割区域
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        # 只要还存在面积小于等于4的分割区域就进行循环过滤
        while filtered_small.any():
            # 根据过滤条件，更新当前的分割区域、分割得分和分割类别
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            # 重新获取分割图像和分割区域的面积
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            # 再次过滤面积小于等于4的分割区域
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果当前类别为空，则创建一个具有默认分类的数组
        cur_classes = np.ones((1, 1), dtype=np.int64)

    # 获取分割区域信息，包括ID、是否为物体、类别ID和面积
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除当前类别数组
    del cur_classes

    # 使用IO流保存分割图像
    with io.BytesIO() as out:
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 构建预测结果，包括PNG图像和分割区域信息
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    # 返回预测结果
    return predictions
def resize_annotation(
    annotation: Dict[str, Any],  # 接受一个包含annotation的字典
    orig_size: Tuple[int, int],   # 原始图像的大小
    target_size: Tuple[int, int],  # 目标大小
    threshold: float = 0.5,       # 阈值用于对分割蒙版进行二值化
    resample: PILImageResampling = PILImageResampling.NEAREST,  # 重新采样时使用的滤波器
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):
            The original size of the input image.
        target_size (`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))  # 计算需要缩放的比例
    ratio_height, ratio_width = ratios  # 获取缩放比例

    new_annotation = {}  # 创建新的annotation字典
    new_annotation["size"] = target_size  # 将目标大小存入新字典

    for key, value in annotation.items():  # 遍历原始annotation的键值对
        if key == "boxes":  # 如果键是“boxes”
            boxes = value  # 获取box的值
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)  # 根据缩放比例对box进行缩放
            new_annotation["boxes"] = scaled_boxes  # 将缩放后的box存入新字典
        elif key == "area":  # 如果键是“area”
            area = value  # 获取面积的值
            scaled_area = area * (ratio_width * ratio_height)  # 根据缩放比例对面积进行缩放
            new_annotation["area"] = scaled_area  # 将缩放后的面积存入新字典
        elif key == "masks":  # 如果键是“masks”
            masks = value[:, None]  # 获取掩码
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])  # 根据缩放比例对掩码进行缩放
            masks = masks.astype(np.float32)  # 将掩码转换为浮点型
            masks = masks[:, 0] > threshold  # 根据阈值进行二值化
            new_annotation["masks"] = masks  # 将处理后的掩码存入新字典
        elif key == "size":  # 如果键是“size”
            new_annotation["size"] = target_size  # 更新新字典中的大小
        else:  # 其他情况
            new_annotation[key] = value  # 将原始值复制到新字典中

    return new_annotation  # 返回新字典


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
    if is_torch_tensor(mask):  # 如果是torch张量
        mask = mask.numpy()  # 转换为numpy数组

    pixels = mask.flatten()  # 将掩码展平为一维数组
    pixels = np.concatenate([[0], pixels, [0]])  # 在掩码数组两侧各拼接一个0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 找到掩码中从0到1或者从1到0的转变点
    runs[1::2] -= runs[::2]  # 计算每个区域的长度
    return list(runs)  # 返回运行长度编码的列表


def convert_segmentation_to_rle(segmentation):
    """
    # 将给定的分割图像(segmentation map)的形状为 (height, width) 转换为游程编码(run-length encoding, RLE)格式。
    def segmentation_to_rle(segmentation):
        # 获取分割图像中唯一的分割ID
        segment_ids = torch.unique(segmentation)
    
        # 初始化一个列表来存储每个分割ID的游程编码
        run_length_encodings = []
        for idx in segment_ids:
            # 根据分割ID生成掩码张量
            mask = torch.where(segmentation == idx, 1, 0)
            # 将掩码张量转换为游程编码
            rle = binary_mask_to_rle(mask)
            # 将该分割ID的游程编码添加到列表中
            run_length_encodings.append(rle)
    
        # 返回包含所有分割ID的游程编码的列表
        return run_length_encodings
# 从transformers.models.detr.image_processing_detr.remove_low_and_no_objects中复制的函数，用于移除低质量和不存在对象的掩码
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    用`object_mask_threshold`对给定的掩码进行二值化，返回`masks`、`scores`和`labels`的关联值。

    Args:
        masks (`torch.Tensor`):
            形状为`(num_queries, height, width)`的张量。
        scores (`torch.Tensor`):
            形状为`(num_queries)`的张量。
        labels (`torch.Tensor`):
            形状为`(num_queries)`的张量。
        object_mask_threshold (`float`):
            用于二值化掩码的0到1之间的数字。
    Raises:
        `ValueError`: 当所有输入张量中的第一个维度不匹配时引发异常。
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: 不包含小于`object_mask_threshold`的区域的`masks`、`scores`和`labels`。
    """

# 从transformers.models.detr.image_processing_detr.check_segment_validity中复制的函数，用于检查段的有效性
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第k类相关联的掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询k中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除断开的微小段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 从transformers.models.detr.image_processing_detr.compute_segments中复制的函数，用于计算段
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

    # 根据其预测得分对每个掩码进行加权
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    # 定义一个空的字典，用于存储物品的内存列表，键为字符串，值为整数
    stuff_memory_list: Dict[str, int] = {}
    # 遍历预测标签的行数
    for k in range(pred_labels.shape[0]):
        # 获取预测的类别
        pred_class = pred_labels[k].item()
        # 判断是否应该进行融合
        should_fuse = pred_class in label_ids_to_fuse
    
        # 检查掩码是否存在并且足够大以成为一个段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )
    
        # 如果掩码存在
        if mask_exists:
            # 如果预测类别在物品内存列表中
            if pred_class in stuff_memory_list:
                # 获取当前段的ID
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 否则，当前段的ID加1
                current_segment_id += 1
    
            # 将当前对象段添加到最终分割地图中
            segmentation[mask_k] = current_segment_id
            # 对段得分进行舍入并保存
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            # 如果应该融合，则更新物品内存列表
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id
    
    # 返回最终的分割地图和段列表
    return segmentation, segments
# 创建一个名为DeformableDetrImageProcessor的类，它是BaseImageProcessor的子类
class DeformableDetrImageProcessor(BaseImageProcessor):
    # 类的文档字符串，描述了类的作用和参数
    r"""
    Constructs a Deformable DETR image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's (height, width) dimensions after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
    """

    # 类属性，指定模型接受的输入名称列表
    model_input_names = ["pixel_values", "pixel_mask"]

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.__init__复制代码
    # 定义一个 DeformableDetrImageProcessor 类，用于处理和预处理图像和目标注释
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION, # 设置默认的注释格式为 COCO 目标检测
        do_resize: bool = True, # 是否进行图像缩放
        size: Dict[str, int] = None, # 指定缩放后的图像尺寸
        resample: PILImageResampling = PILImageResampling.BILINEAR, # 设置缩放时使用的采样方法
        do_rescale: bool = True, # 是否进行图像像素值范围缩放
        rescale_factor: Union[int, float] = 1 / 255, # 指定图像像素值缩放因子
        do_normalize: bool = True, # 是否进行图像归一化
        image_mean: Union[float, List[float]] = None, # 指定图像归一化时使用的均值
        image_std: Union[float, List[float]] = None, # 指定图像归一化时使用的标准差
        do_pad: bool = True, # 是否进行图像填充
        **kwargs,
    ) -> None:
        # 处理 kwargs 中的一些特殊参数
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")
    
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None if size is None else 1333
    
        # 设置图像尺寸
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
    
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
    
    @classmethod
    # 重写父类的 from_dict 方法，确保在使用 from_dict 创建图像处理器时能够正确处理参数
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        image_processor_dict = image_processor_dict.copy()
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        return super().from_dict(image_processor_dict, **kwargs)
    
    # 重写父类的 prepare_annotation 方法，用于预处理目标注释
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 在这里实现预处理目标注释的逻辑
        pass
    # 定义一个函数，准备输入到 DeformableDetr 模型的注释
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None) -> Dict:
        """
        Prepare an annotation for feeding into DeformableDetr model.
        """
        # 如果格式不为空，则采用指定格式；否则使用默认格式
        format = format if format is not None else self.format
    
        # 如果格式为 COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果返回分割掩码为空，则设为 False；否则使用指定值
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO_DETECTION 的注释
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果格式为 COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果返回分割掩码为空，则设为 True；否则使用指定值
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO_PANOPTIC 的注释
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 抛出数值错误，指出不支持的格式
            raise ValueError(f"Format {format} is not supported.")
        # 返回处理好的注释
        return target
    
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare 复制而来
    # 该方法已弃用，将在 v4.33 版本中删除
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用新的方法 prepare_annotation 处理注释
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return image, target
    
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.convert_coco_poly_to_mask 复制而来
    # 该方法已弃用，将在 v4.33 版本中删除
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用新的方法 convert_coco_poly_to_mask 处理注释
        return convert_coco_poly_to_mask(*args, **kwargs)
    
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection 复制而来
    # 该方��已弃用，将在 v4.33 版本中删除
    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用新的方法 prepare_coco_detection_annotation 处理注释
        return prepare_coco_detection_annotation(*args, **kwargs)
    
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic 复制而来
    # 该方法已弃用，将在 v4.33 版本中删除
    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用新的方法 prepare_coco_panoptic_annotation 处理注释
        return prepare_coco_panoptic_annotation(*args, **kwargs)
    
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.resize 复制而来
    # 定义resize方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        调整图像大小到给定的尺寸。尺寸可以是`min_size`（标量）或`(height, width)`元组。 如果尺寸是整数，则图像的较小边将与此数字匹配。

        Args:
            image (`np.ndarray`):
                要调整大小的图像。
            size (`Dict[str, int]`):
                包含要调整大小的尺寸的字典。可以包含键`shortest_edge`和`longest_edge`或`height`和`width`。
            resample (`PILImageResampling`, *optional*, 默认为`PILImageResampling.BILINEAR`):
                在调整图像大小时要使用的重采样滤波器。
            data_format (`str` or `ChannelDimension`, *optional*):
                输出图像的通道维度格式。 如果未设置，则使用输入图像的通道维度格式。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。 如果未提供，将进行推断。
        """
        # 检查是否存在已弃用的`max_size`参数，如果存在则发出警告
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # 获取真实的尺寸字典，根据`size`和`max_size`，默认不生成正方形图像
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        # 根据尺寸字典获取调整大小后的图像尺寸
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果尺寸字典不含有所需的键，则引发错误
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # 执行图像调整大小操作
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        # 返回调整大小后的图像
        return image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation中复制的方法
    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    def resize_annotation(
        self,
        annotation: Dict,
        orig_size: Tuple[int, int],
        size: Union[int, Tuple[int, int]],
        resample: int = Image.NEAREST,
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

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

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format.
        """
        return normalize_annotation(annotation, image_size=image_size)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad the image to the given output size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            output_size (`Tuple[int, int]`):
                Target size of the output image (height, width).
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value or values to pad with. If a single `float` is provided, it is used for all channels. If
                multiple values are provided, their order should match the number of channels in the image.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image.
        """
    # 定义一个函数，用于将图像用零填充到指定大小
    def pad(
        self,
        images: List[np.ndarray],  # 输入图像列表
        constant_values: Union[float, Iterable[float]] = 0,  # 填充值，默认为0
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 是否返回张量，默认为None
        data_format: Optional[ChannelDimension] = None,  # 数据格式，可以为空
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可以为空
    ) -> np.ndarray:  # 返回类型为np.ndarray
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
        # 设置填充方式为((0, pad_bottom), (0, pad_right))
        padding = ((0, pad_bottom), (0, pad_right))
        # 使用指定的填充方式和值对图像进行填充
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,  # 填充模式为常数填充
            constant_values=constant_values,  # 指定常数填充的值
            data_format=data_format,  # 指定数据格式
            input_data_format=input_data_format,  # 指定输入数据格式
        )
        # 返回填充后的图像
        return padded_image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.pad复制而来
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
        # 计算需要填充的大小，使所有图像都能达到批次中最大高度和宽度
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对批次中的每张图像进行填充
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
        # 构建数据字典，包含填充后的图像像素值
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 为每张图像生成像素掩码
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 将像素掩码添加到数据字典中
            data["pixel_mask"] = masks

        # 返回批次特征对象，包含填充后的数据和张量类型
        return BatchFeature(data=data, tensor_type=return_tensors)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.preprocess复制而来
    # 预处理方法，用于对输入数据进行预处理操作
    def preprocess(
        self,
        # 图像输入，可以是 PIL Image 或路径
        images: ImageInput,
        # 注释，可以是单个注释或注释列表，默认为 None
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        # 是否返回分割掩模
        return_segmentation_masks: bool = None,
        # 分割掩模路径，可以是字符串或路径，默认为 None
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        # 是否进行调整大小
        do_resize: Optional[bool] = None,
        # 图像大小，包含宽高信息，默认为 None
        size: Optional[Dict[str, int]] = None,
        # 重采样方法，PIL ImageResampling，默认为 None
        resample=None,
        # 是否进行缩放
        do_rescale: Optional[bool] = None,
        # 缩放因子，可以是整数或浮点数，默认为 None
        rescale_factor: Optional[Union[int, float]] = None,
        # 是否进行归一化
        do_normalize: Optional[bool] = None,
        # 图像均值，可以是单个值或列表，默认为 None
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差，可以是单个值或列表，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否进行填充
        do_pad: Optional[bool] = None,
        # 注释格式，可以是字符串或注释格式，默认为 None
        format: Optional[Union[str, AnnotationFormat]] = None,
        # 返回张量类型，可以是张量类型或字符串，默认为 None
        return_tensors: Optional[Union[TensorType, str]] = None,
        # 数据格式，通道位置，第一维或者其他，默认为 ChannelDimension.FIRST
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        # 输入数据格式，可以是字符串或通道位置，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
    # 后处理方法 - 待完成: 添加对其他框架的支持
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 发出警告，提醒`post_process`将在Transformers v5中被移除，建议使用`post_process_object_detection`代替，并将阈值设置为0以获得相同的结果
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 从模型输出中获取预测的logits和边界框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查logits和target_sizes的维度是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查target_sizes的每个元素是否包含批次中每个图像的尺寸(h, w)
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 将logits转换为概率值
        prob = out_logits.sigmoid()
        # 获取top k的logits值和对应的索引
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        # 将边界框转换为顶点坐标表示
        boxes = center_to_corners_format(out_bbox)
        # 通过索引获取top k的边界框
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 将相对坐标[0, 1]转换为绝对坐标[0, height]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 将结果整理成字典列表
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None, top_k: int = 100
        """
        将[`DeformableDetrForObjectDetection`]的原始输出转换为最终的边界框，格式为(top_left_x, top_left_y, bottom_right_x, bottom_right_y)。仅支持PyTorch。

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                模型的原始输出。
            threshold (`float`, *optional*):
                用于保留目标检测预测的得分阈值。
            target_sizes (`torch.Tensor` 或 `List[Tuple[int, int]]`, *optional*):
                形状为`(batch_size, 2)`的张量或包含每个图像批次中目标大小（高度，宽度）的元组列表 (`Tuple[int, int]`)。
                如果为None，则不调整预测大小。
            top_k (`int`, *optional*, 默认为100):
                在通过阈值过滤之前仅保留前k个边界框。

        Returns:
            `List[Dict]`: 一个字典列表，每个字典包含模型预测的图像批次中的得分、标签和框。
        """
        # 从输出中获取逻辑输出和预测框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 如果指定了目标大小
        if target_sizes is not None:
            # 确保目标大小的数量与逻辑输出的批次维度相匹配
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # 对逻辑输出进行sigmoid处理
        prob = out_logits.sigmoid()
        # 重新塑造概率张量
        prob = prob.view(out_logits.shape[0], -1)
        # 确定要保留的top_k值和索引
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        # 提取得分和标签
        scores = topk_values
        labels = topk_indexes % out_logits.shape[2]
        # 将预测框转换为以中心为基准的坐标格式
        boxes = center_to_corners_format(out_bbox)
        # 使用topk索引提取对应的边界框
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 将相对[0, 1]的坐标转换为绝对[0, height]的坐标
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        # 遍历每个图像的得分、标签和框
        for s, l, b in zip(scores, labels, boxes):
            # 根据阈值过滤得分、标签和框
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        # 返回结果
        return results
```