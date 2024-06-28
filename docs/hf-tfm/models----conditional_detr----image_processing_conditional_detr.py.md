# `.\models\conditional_detr\image_processing_conditional_detr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及版权许可信息
# 此代码基于 Apache License, Version 2.0 许可
# 详细信息请访问 http://www.apache.org/licenses/LICENSE-2.0

"""Conditional DETR 的图像处理器类。"""

# 导入所需库和模块
import io  # 提供了对 I/O 操作的支持
import pathlib  # 提供了操作文件和目录路径的功能
from collections import defaultdict  # 提供了默认值的字典实现
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union  # 引入类型提示

import numpy as np  # 引入 NumPy 数学库，用于数组操作

# 从 HuggingFace 库中导入图像处理相关的工具和模块
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
    validate_kwargs,
    validate_preprocess_arguments,
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

# 如果 Torch 可用，导入 Torch 相关模块
if is_torch_available():
    import torch
    from torch import nn

# 如果 vision 相关工具可用，导入 PIL 库
if is_vision_available():
    import PIL

# 如果 SciPy 可用，导入 SciPy 的特定模块
if is_scipy_available():
    import scipy.special
    import scipy.stats

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，用于日志记录


SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)

# 从 transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio 复制而来
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    根据输入图像大小和所需输出大小计算输出图像的尺寸。

    Args:
        image_size (`Tuple[int, int]`):
            输入图像的尺寸.
        size (`int`):
            所需的输出尺寸.
        max_size (`int`, *optional*):
            允许的最大输出尺寸.

    Returns:
        Tuple[int, int]: 输出图像的高度和宽度.
    """
    height, width = image_size
    # 如果指定了最大尺寸限制
    if max_size is not None:
        # 计算原始尺寸的最小值和最大值
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        # 如果根据原始尺寸计算的新尺寸超过了最大限制，则调整尺寸大小
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    
    # 如果图像高度小于等于宽度且高度等于指定尺寸，或者宽度小于等于高度且宽度等于指定尺寸，则直接返回原始尺寸
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    
    # 根据图像宽高比例计算新的缩放后的宽度和高度
    if width < height:
        ow = size  # 新的宽度为指定尺寸
        oh = int(size * height / width)  # 根据比例计算新的高度
    else:
        oh = size  # 新的高度为指定尺寸
        ow = int(size * width / height)  # 根据比例计算新的宽度
    
    return (oh, ow)  # 返回新的图像尺寸元组
# Copied from transformers.models.detr.image_processing_detr.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
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
    # 获取输入图片的尺寸
    image_size = get_image_size(input_image, input_data_format)
    # 如果输出大小是一个 tuple 或 list，则直接返回
    if isinstance(size, (list, tuple)):
        return size
    # 否则，根据输入图片尺寸和指定的大小计算保持宽高比的输出图片尺寸
    return get_size_with_aspect_ratio(image_size, size, max_size)


# Copied from transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    # 如果输入是 numpy 数组，则返回 numpy 的数组转换函数
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果 TensorFlow 可用且输入是 TensorFlow 的张量，则返回 TensorFlow 的转换函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果 PyTorch 可用且输入是 PyTorch 的张量，则返回 PyTorch 的转换函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果 Flax 可用且输入是 JAX 的张量，则返回 JAX 的数组转换函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 如果以上都不是，则抛出值错误异常
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# Copied from transformers.models.detr.image_processing_detr.safe_squeeze
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    # 如果未指定轴，则直接调用 squeeze 方法
    if axis is None:
        return arr.squeeze()
    # 否则，尝试在指定轴上调用 squeeze 方法，如果出现值错误则返回原数组
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# Copied from transformers.models.detr.image_processing_detr.normalize_annotation
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    norm_annotation = {}
    # 遍历注释字典中的每个键值对
    for key, value in annotation.items():
        # 如果当前键是"boxes"
        if key == "boxes":
            # 将值赋给变量boxes，并将边角坐标格式转换为中心坐标格式
            boxes = value
            boxes = corners_to_center_format(boxes)
            # 将坐标值除以图像宽度和高度，以归一化坐标
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的坐标存入norm_annotation字典
            norm_annotation[key] = boxes
        else:
            # 对于除了"boxes"以外的其他键，直接复制其值到norm_annotation字典中
            norm_annotation[key] = value
    # 返回归一化后的注释字典
    return norm_annotation
# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    # 使用 zip(*values) 对传入的可迭代对象进行解压，获取每个索引位置上的值集合
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未指定输入数据格式，则通过第一张图像推断通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据输入数据格式的不同，计算图像中的最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        # 如果输入数据格式不是有效的通道维度格式，则引发异常
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
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
    # 获取图像的输入高度和宽度，根据数据格式不同可能会进行转换
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    
    # 创建一个大小为 output_size 的像素掩码，初始值全部为 0
    mask = np.zeros(output_size, dtype=np.int64)
    
    # 将有效像素位置（即图像的实际尺寸）标记为 1
    mask[:input_height, :input_width] = 1
    return mask


# Copied from transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask
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
        # 尝试导入 pycocotools 的 mask 模块
        from pycocotools import mask as coco_mask
    except ImportError:
        # 如果导入失败，则抛出 ImportError 异常
        raise ImportError("Pycocotools is not installed in your environment.")

    # 初始化一个空列表来存储所有的掩码
    masks = []
    
    # 遍历每个多边形的坐标列表，将其转换为 COCO 格式的 RLE 编码，再解码为二进制掩码
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        
        # 如果掩码的维度少于 3（缺少颜色通道维度），则添加一个额外的维度
        if len(mask.shape) < 3:
            mask = mask[..., None]
        
        # 将掩码转换为 uint8 类型的 numpy 数组，并将所有非零值转换为 True（1），零值转换为 False（0）
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)  # 将所有颜色通道的信息合并成一个单一的掩码
        
        # 将生成的掩码添加到掩码列表中
        masks.append(mask)
    
    # 如果成功生成了掩码列表，则将它们堆叠成一个 numpy 数组返回；否则返回一个空的掩码数组
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks
# 从transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation复制，将DETR转换为ConditionalDetr
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    将COCO格式的目标转换为ConditionalDetr所期望的格式。
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取图像ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取所有给定图像的COCO注释
    annotations = target["annotations"]
    # 过滤掉"iscrowd"属性为1的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有目标的类别ID
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 用于转换为COCO API格式
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取所有目标的边界框
    boxes = [obj["bbox"] for obj in annotations]
    # 防止没有边界框时通过调整大小来处理
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 确保边界框的有效性
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果注释中包含关键点信息
    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将筛选后的关键点列表转换为numpy数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 在此应用keep掩码以筛选相关的注释
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码
    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


# 从transformers.models.detr.image_processing_detr.masks_to_boxes复制
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩码周围的边界框。

    Args:
        masks: 格式为`[number_masks, height, width]`的掩码，其中N是掩码数量

    Returns:
        boxes: 格式为`[number_masks, 4]`的边界框，xyxy格式
    """
    # 如果masks数组为空，返回一个形状为(0, 4)的零数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取masks数组的高度h和宽度w
    h, w = masks.shape[-2:]
    
    # 创建一个包含0到h-1的浮点数数组y，和一个包含0到w-1的浮点数数组x
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    
    # 创建y和x的网格，使用"ij"索引顺序，详见https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    # 将masks数组与x的扩展维度相乘，计算每个像素在x轴上的最大值和最小值
    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    
    # 使用masks数组的布尔反转掩码创建一个掩码数组x，填充填充值为1e8
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 将masks数组与y的扩展维度相乘，计算每个像素在y轴上的最大值和最小值
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    
    # 使用masks数组的布尔反转掩码创建一个掩码数组y，填充填充值为1e8
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 将x_min、y_min、x_max和y_max堆叠成一个形状为(?, 4)的数组并返回
    return np.stack([x_min, y_min, x_max, y_max], 1)
# Copied from transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation with DETR->ConditionalDetr
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for ConditionalDetr.
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 构建注释文件的路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    # 设置图像ID
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 设置图像大小
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 设置原始图像大小
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        # 读取注释文件中的掩码图像并转换成numpy数组
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将RGB格式的掩码图像转换成ID格式
        masks = rgb_to_id(masks)

        # 获取所有分段信息中的ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 生成掩码数组，表示每个像素属于哪个分段
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        
        if return_masks:
            # 如果需要返回掩码，则将其存储在new_target中
            new_target["masks"] = masks
        # 将掩码转换为边界框
        new_target["boxes"] = masks_to_boxes(masks)
        # 存储每个分段的类别标签
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 存储每个分段的is_crowd标志
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 存储每个分段的面积信息
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


# Copied from transformers.models.detr.image_processing_detr.get_segmentation_image
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入图像的高度和宽度
    h, w = input_size
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size

    # 对掩码进行softmax操作，并转置
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    if m_id.shape[-1] == 0:
        # 如果未检测到任何掩码，则生成全零掩码
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取最大概率的掩码ID，并重新整形为输入图像大小
        m_id = m_id.argmax(-1).reshape(h, w)

    if deduplicate:
        # 如果需要去除重复掩码，则将属于相同类别的掩码ID合并为一个
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将掩码ID转换成RGB格式的分割图像
    seg_img = id_to_rgb(m_id)
    # 将分割图像调整到最终的目标尺寸
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# Copied from transformers.models.detr.image_processing_detr.get_mask_area
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size
    # 将分割图像转换成numpy数组，并设置数据类型为uint8
    np_seg_img = seg_img.astype(np.uint8)
    # 调整numpy数组形状以匹配最终的目标尺寸
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 使用 rgb_to_id 函数将 np_seg_img 转换为标识图像 m_id
    m_id = rgb_to_id(np_seg_img)
    # 计算每个类别的像素数量，存储在列表 area 中
    area = [(m_id == i).sum() for i in range(n_classes)]
    # 返回计算得到的像素数量列表 area
    return area
# Copied from transformers.models.detr.image_processing_detr.score_labels_from_class_probabilities
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 计算类别概率的 softmax，将 logits 转换为概率值
    probs = scipy.special.softmax(logits, axis=-1)
    # 获取每个样本的预测类别
    labels = probs.argmax(-1, keepdims=True)
    # 根据预测类别从概率数组中取得对应的分数
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 去除多余的维度，得到一维数组的分数和类别
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# Copied from transformers.models.detr.image_processing_detr.post_process_panoptic_sample with DetrForSegmentation->ConditionalDetrForSegmentation
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
    Converts the output of [`ConditionalDetrForSegmentation`] into panoptic segmentation predictions for a single sample.

    Args:
        out_logits (`torch.Tensor`):
            The logits for this sample.
        masks (`torch.Tensor`):
            The predicted segmentation masks for this sample.
        boxes (`torch.Tensor`):
            The prediced bounding boxes for this sample. The boxes are in the normalized format `(center_x, center_y,
            width, height)` and values between `[0, 1]`, relative to the size the image (disregarding padding).
        processed_size (`Tuple[int, int]`):
            The processed size of the image `(height, width)`, as returned by the preprocessing step i.e. the size
            after data augmentation but before batching.
        target_size (`Tuple[int, int]`):
            The target size of the image, `(height, width)` corresponding to the requested final size of the
            prediction.
        is_thing_map (`Dict`):
            A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.
        threshold (`float`, *optional*, defaults to 0.85):
            The threshold used to binarize the segmentation masks.
    """
    # 过滤掉预测类别为无效类别或分数低于阈值的结果
    scores, labels = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    cur_masks = masks[keep]
    # 将预测的分割掩码调整大小以适应处理后的图像尺寸
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # 可能存在同一类别的多个预测掩码，此处跟踪每个物体类别的掩码 ID 列表（稍后将合并）
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    # 遍历当前类别列表及其对应的标签
    for k, label in enumerate(cur_classes):
        # 如果当前标签不是物体类别，则将其索引添加到对应的等效类别列表中
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    # 根据当前的掩膜图像获取分割图像
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    # 获取当前掩膜的面积
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # 过滤掉面积过小的掩膜
    if cur_classes.size() > 0:
        # 创建布尔数组，标记面积小于等于4的掩膜
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        # 只要还有面积小于等于4的掩膜，就循环过滤
        while filtered_small.any():
            # 从当前掩膜中移除面积过小的掩膜
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            # 根据更新后的掩膜获取分割图像
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            # 获取更新后的掩膜的面积
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            # 再次过滤面积小于等于4的掩膜
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果当前类别列表为空，则创建一个包含单一元素的整数数组
        cur_classes = np.ones((1, 1), dtype=np.int64)

    # 创建分段信息列表，包含每个分段的ID、是否为物体、类别ID及其面积
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除不再需要的当前类别数组
    del cur_classes

    # 创建一个字节流对象
    with io.BytesIO() as out:
        # 将分割图像转换为PIL图像并保存为PNG格式
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 构造预测结果字典，包含PNG格式的字符串及分段信息
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    # 返回最终预测结果字典
    return predictions
# Copied from transformers.models.detr.image_processing_detr.resize_annotation
def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
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
    # 计算尺寸比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    # 创建一个新的注释字典，将目标大小放入其中
    new_annotation = {}
    new_annotation["size"] = target_size

    # 遍历原始注释的键值对
    for key, value in annotation.items():
        if key == "boxes":
            # 对象边界框的调整尺寸
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes
        elif key == "area":
            # 目标区域的调整尺寸
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area
        elif key == "masks":
            # 掩码的调整尺寸
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            new_annotation["masks"] = masks
        elif key == "size":
            # 更新目标大小（可能被覆盖）
            new_annotation["size"] = target_size
        else:
            # 复制其它键值对
            new_annotation[key] = value

    # 返回调整尺寸后的新注释字典
    return new_annotation


# Copied from transformers.models.detr.image_processing_detr.binary_mask_to_rle
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
    # 如果输入是 PyTorch 张量，则转换为 NumPy 数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将二进制掩码展平为一维数组
    pixels = mask.flatten()
    # 添加额外的零到像素数组的两端
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到像素变化的位置索引
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 计算 RLE 编码长度
    runs[1::2] -= runs[::2]
    # 返回运行长度编码列表
    return list(runs)


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    # 获取分割图中唯一的分割标识符（segmentation id）
    segment_ids = torch.unique(segmentation)
    
    # 初始化用于存储所有分割标识符的运行长度编码的列表
    run_length_encodings = []
    
    # 遍历每个分割标识符
    for idx in segment_ids:
        # 创建一个掩码，其中分割图中与当前标识符相同的位置为1，否则为0
        mask = torch.where(segmentation == idx, 1, 0)
        
        # 将二进制掩码转换为运行长度编码（RLE）
        rle = binary_mask_to_rle(mask)
        
        # 将当前标识符的运行长度编码添加到结果列表中
        run_length_encodings.append(rle)
    
    # 返回所有分割标识符的运行长度编码列表
    return run_length_encodings
# Copied from transformers.models.detr.image_processing_detr.compute_segments

# 根据目标大小或默认大小计算掩码的高度和宽度
height = mask_probs.shape[1] if target_size is None else target_size[0]
width = mask_probs.shape[2] if target_size is None else target_size[1]

# 创建一个与图像大小相同的空白分割结果张量，用于存储每个像素点的分割标识
segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
# 初始化空的分割结果列表，用于存储每个分割的详细信息
segments: List[Dict] = []

# 如果设置了目标大小，则对掩码进行插值以适应目标大小
if target_size is not None:
    mask_probs = nn.functional.interpolate(
        mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    )[0]

# 当前分割的 ID
current_segment_id = 0

# 根据预测分数加权每个掩码
mask_probs *= pred_scores.view(-1, 1, 1)
# 找到每个像素点最可能的类别标签
mask_labels = mask_probs.argmax(0)  # [height, width]

# 用于跟踪每个类别的实例数量
    # 初始化一个空的字典，用于存储每个类别的对象段的内存索引
    stuff_memory_list: Dict[str, int] = {}
    
    # 遍历预测标签的每一行
    for k in range(pred_labels.shape[0]):
        # 获取当前预测类别的整数表示
        pred_class = pred_labels[k].item()
    
        # 检查当前预测类别是否需要融合
        should_fuse = pred_class in label_ids_to_fuse
    
        # 检查当前索引处的掩码是否存在并且足够大以表示一个段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )
    
        # 如果存在有效的掩码
        if mask_exists:
            # 如果当前预测类别已经在内存列表中存在
            if pred_class in stuff_memory_list:
                # 获取当前预测类别的段的内存索引
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 如果当前预测类别不在内存列表中，则增加段的内存索引
                current_segment_id += 1
    
            # 将当前对象段添加到最终的分割映射中
            segmentation[mask_k] = current_segment_id
    
            # 获取当前段的预测得分并进行四舍五入保留小数点后六位
            segment_score = round(pred_scores[k].item(), 6)
    
            # 将当前段的信息添加到段列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
    
            # 如果需要融合，则更新内存列表中当前预测类别的段的内存索引
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id
    
    # 返回最终的分割映射和段列表
    return segmentation, segments
class ConditionalDetrImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Conditional Detr image processor.

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
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True` will pad the images in the batch to the largest height and width in the batch.
            Padding will be applied to the bottom and right of the image with zeros.
    """
    model_input_names = ["pixel_values", "pixel_mask"]

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.__init__
    # 初始化函数，用于创建一个图像处理器对象
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_annotations: Optional[bool] = None,
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        # 如果 kwargs 中有 "pad_and_return_pixel_mask"，则设置 do_pad 为其值并移除该参数
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 如果 kwargs 中有 "max_size"，则发出警告提示并将其移除，推荐使用 size['longest_edge']
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None if size is None else 1333

        # 如果 size 为 None，则设置默认 size 字典
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # 根据 size 和 max_size 获取最终的尺寸字典
        size = get_size_dict(size, max_size=max_size, default_to_square=False)

        # 向父类初始化方法传递其余的 kwargs 参数
        super().__init__(**kwargs)
        # 设置各个属性值
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_annotations = do_convert_annotations
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        # 定义一个有效的处理器键名列表，用于验证和配置处理器参数
        self._valid_processor_keys = [
            "images",
            "annotations",
            "return_segmentation_masks",
            "masks_path",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "do_convert_annotations",
            "image_mean",
            "image_std",
            "do_pad",
            "format",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.from_dict with Detr->ConditionalDetr
    # 重写基类中的 `from_dict` 方法，用于从字典创建 ConditionalDetrImageProcessor 对象，
    # 并确保在使用 from_dict 方法创建图像处理器时更新参数，例如 `ConditionalDetrImageProcessor.from_pretrained(checkpoint, size=600, max_size=800)`
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        # 复制输入的 image_processor_dict，以确保不修改原始字典
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中有 "max_size" 参数，则更新到 image_processor_dict 中，并从 kwargs 中移除该参数
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果 kwargs 中有 "pad_and_return_pixel_mask" 参数，则更新到 image_processor_dict 中，并从 kwargs 中移除该参数
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        # 调用父类的 from_dict 方法，使用更新后的 image_processor_dict 和任何额外的 kwargs 参数
        return super().from_dict(image_processor_dict, **kwargs)

    # 从 DETR 源码中复制的方法，准备输入图像的注释，以便供 ConditionalDetr 模型使用
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into ConditionalDetr model.
        """
        # 如果未指定 format，则使用对象中存储的 format
        format = format if format is not None else self.format

        # 如果 format 是 AnnotationFormat.COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果未指定 return_segmentation_masks，则设为 False；否则保持原值
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_detection_annotation 方法，准备 COCO 检测格式的注释
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果 format 是 AnnotationFormat.COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果未指定 return_segmentation_masks，则设为 True；否则保持原值
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_panoptic_annotation 方法，准备 COCO 全景格式的注释
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 如果 format 不是支持的格式，则抛出 ValueError 异常
            raise ValueError(f"Format {format} is not supported.")
        
        # 返回处理后的 target 字典
        return target

    # 从 DETR 源码中复制的方法，准备输入数据，调用 prepare_annotation 方法处理注释
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        # 发出一次性警告，提示 `prepare` 方法已弃用，并将在 v4.33 版本中移除
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用 prepare_annotation 方法处理输入的 image 和 target，并返回处理后的 image 和 target
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return image, target

    # 从 DETR 源码中复制的方法，将 COCO 多边形转换为掩码的方法，未完成复制
    # 发出警告日志，提示`convert_coco_poly_to_mask`方法已弃用，将在v4.33版本移除
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用同名函数`convert_coco_poly_to_mask`并返回其结果
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection`复制而来，已经修改了`DETR`为`ConditionalDetr`
    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用`prepare_coco_detection_annotation`函数并返回其结果
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic`复制而来
    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用`prepare_coco_panoptic_annotation`函数并返回其结果
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.resize`复制而来，定义了图像的调整大小函数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 定义函数 `resize_image`，接受 `image` 和 `size` 参数，返回 `np.ndarray` 类型
    def resize_image(
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[str] = None,
        input_data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ):
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
        # 检查 `kwargs` 中是否包含 `max_size` 参数，如果包含则发出警告
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            # 弹出 `kwargs` 中的 `max_size` 参数，并赋值给 `max_size` 变量
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 调用 `get_size_dict` 函数，根据 `size` 和 `max_size` 参数获取尺寸字典 `size`
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 检查 `size` 字典中是否同时包含 `shortest_edge` 和 `longest_edge` 键
        if "shortest_edge" in size and "longest_edge" in size:
            # 调用 `get_resize_output_image_size` 函数，根据 `shortest_edge` 和 `longest_edge` 获取调整后的尺寸
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        # 否则，检查 `size` 字典中是否同时包含 `height` 和 `width` 键
        elif "height" in size and "width" in size:
            # 直接取出 `height` 和 `width` 的值，组成元组赋值给 `size`
            size = (size["height"], size["width"])
        else:
            # 如果 `size` 字典中既不包含 `shortest_edge` 和 `longest_edge`，也不包含 `height` 和 `width`，则抛出数值错误异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 调用 `resize` 函数，根据给定参数调整图像大小，并将结果赋值给 `image`
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        
        # 返回调整大小后的图像
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation
    # 定义 `resize_annotation` 方法，接受 `annotation`、`orig_size`、`size` 和可选的 `resample` 参数，返回 `Dict` 类型
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
        # 调用 `resize_annotation` 函数，根据参数调整注释大小，并返回结果
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    # 使用给定的因子对图像进行重新缩放，即 image = image * rescale_factor。
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
        # 调用全局函数 rescale，对图像进行重新缩放
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation 复制而来
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        # 调用全局函数 normalize_annotation，规范化注释中的边界框格式
        return normalize_annotation(annotation, image_size=image_size)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor._update_annotation_for_padded_image 复制而来
    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ):
        """
        Update annotations to account for padding in the image.

        Args:
            annotation (`Dict`):
                Dictionary containing annotations.
            input_image_size (`Tuple[int, int]`):
                Original size of the input image (height, width).
            output_image_size (`Tuple[int, int]`):
                Size of the output image after padding (height, width).
            padding:
                Padding information.
            update_bboxes:
                Whether to update bounding boxes or not.
        """
        # 调用全局函数 _update_annotation_for_padded_image，更新由于图像填充而改变的注释信息
        pass  # 这里使用 pass 表示该方法暂不执行任何操作
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        # 创建一个新的空注释字典
        new_annotation = {}
        # 将输出图像的尺寸信息添加到新注释字典中
        new_annotation["size"] = output_image_size

        # 遍历给定的注释字典
        for key, value in annotation.items():
            # 如果键是 "masks"
            if key == "masks":
                # 获取 masks，并使用 pad 函数进行填充操作
                masks = value
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                # 对填充后的 masks 进行 squeeze 操作，移除大小为 1 的维度
                masks = safe_squeeze(masks, 1)
                # 将填充后的 masks 更新到新注释字典中
                new_annotation["masks"] = masks
            # 如果键是 "boxes" 并且 update_bboxes 为 True
            elif key == "boxes" and update_bboxes:
                # 获取 boxes，并根据输入和输出图像尺寸的比例进行缩放
                boxes = value
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                # 将缩放后的 boxes 更新到新注释字典中
                new_annotation["boxes"] = boxes
            # 如果键是 "size"
            elif key == "size":
                # 将输出图像的尺寸信息更新到新注释字典中
                new_annotation["size"] = output_image_size
            else:
                # 对于其它所有情况，直接将原始值添加到新注释字典中
                new_annotation[key] = value
        
        # 返回更新后的注释字典
        return new_annotation

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        annotation: Optional[Dict[str, Any]] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        update_bboxes: bool = True,
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
        # 构造填充元组
        padding = ((0, pad_bottom), (0, pad_right))
        # 使用 pad 函数对图像进行填充操作
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        
        # 如果提供了注释信息，则更新注释以适应填充后的图像
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        
        # 返回填充后的图像和更新后的注释（如果提供了注释）
        return padded_image, annotation

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 定义一个方法 `pad`，属于当前类的实例方法（self 指向当前对象）
    def pad(
        self,
        images: List[np.ndarray],  # images 参数是一个包含 np.ndarray 元素的列表
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # annotations 参数可选，可以是单个 AnnotationType 或其列表
        constant_values: Union[float, Iterable[float]] = 0,  # constant_values 参数可以是单个浮点数或浮点数的可迭代对象，默认值为 0
        return_pixel_mask: bool = True,  # return_pixel_mask 参数是一个布尔值，默认为 True
        return_tensors: Optional[Union[str, TensorType]] = None,  # return_tensors 参数可选，可以是字符串或 TensorType 类型
        data_format: Optional[ChannelDimension] = None,  # data_format 参数可选，可以是 ChannelDimension 类型
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # input_data_format 参数可选，可以是字符串或 ChannelDimension 类型
        update_bboxes: bool = True,  # update_bboxes 参数是一个布尔值，默认为 True
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.preprocess 复制而来的方法定义
    def preprocess(
        self,
        images: ImageInput,  # images 参数是 ImageInput 类型
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # annotations 参数可选，可以是单个 AnnotationType 或其列表
        return_segmentation_masks: bool = None,  # return_segmentation_masks 参数是一个布尔值，默认为 None
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # masks_path 参数可选，可以是字符串或 pathlib.Path 类型
        do_resize: Optional[bool] = None,  # do_resize 参数可选，可以是布尔值或者 None
        size: Optional[Dict[str, int]] = None,  # size 参数可选，是一个字典，键是字符串，值是整数
        resample=None,  # resample 参数，默认为 None，应为 PILImageResampling 类型
        do_rescale: Optional[bool] = None,  # do_rescale 参数可选，可以是布尔值或者 None
        rescale_factor: Optional[Union[int, float]] = None,  # rescale_factor 参数可选，可以是整数或浮点数
        do_normalize: Optional[bool] = None,  # do_normalize 参数可选，可以是布尔值或者 None
        do_convert_annotations: Optional[bool] = None,  # do_convert_annotations 参数可选，可以是布尔值或者 None
        image_mean: Optional[Union[float, List[float]]] = None,  # image_mean 参数可选，可以是单个浮点数或浮点数的列表
        image_std: Optional[Union[float, List[float]]] = None,  # image_std 参数可选，可以是单个浮点数或浮点数的列表
        do_pad: Optional[bool] = None,  # do_pad 参数可选，可以是布尔值或者 None
        format: Optional[Union[str, AnnotationFormat]] = None,  # format 参数可选，可以是字符串或 AnnotationFormat 类型
        return_tensors: Optional[Union[TensorType, str]] = None,  # return_tensors 参数可选，可以是 TensorType 类型或字符串
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # data_format 参数是字符串或 ChannelDimension 类型，默认为 ChannelDimension.FIRST
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # input_data_format 参数可选，可以是字符串或 ChannelDimension 类型
        **kwargs,  # 其余未命名参数，以字典形式接收
    # 后处理方法 - TODO: 添加对其它框架的支持
    # 将模型输出转换为 Pascal VOC 格式（xmin, ymin, xmax, ymax），该方法已被弃用，建议使用 `post_process_object_detection` 方法代替
    def post_process(self, outputs, target_sizes):
        """
        Converts the output of [`ConditionalDetrForObjectDetection`] into the format expected by the Pascal VOC format (xmin, ymin, xmax, ymax).
        Only supports PyTorch.

        Args:
            outputs ([`ConditionalDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 发出警告信息，提醒用户方法即将在 Transformers v5 中移除，建议使用 `post_process_object_detection` 替代
        logging.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 提取模型输出中的逻辑回归结果和预测框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出的数量与目标大小数量是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标大小的形状是否正确
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对逻辑回归结果进行 sigmoid 激活
        prob = out_logits.sigmoid()

        # 获取概率最高的前 300 个预测结果的值和索引
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        # 计算预测框的索引
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        # 计算预测结果的标签
        labels = topk_indexes % out_logits.shape[2]
        # 将预测框格式从中心点转换为角点坐标格式
        boxes = center_to_corners_format(out_bbox)
        # 根据预测框的索引从所有预测框中选择特定的预测框
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 将相对坐标 [0, 1] 转换为绝对坐标 [0, height]，其中 height 和 width 分别是图像的高度和宽度
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 将结果组织成字典列表，每个字典包含模型对批次中每个图像的预测结果（分数、标签和框）
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        # 返回结果列表
        return results

    # 从 transformers.models.deformable_detr.image_processing_deformable_detr.DeformableDetrImageProcessor.post_process_object_detection 复制并修改为 ConditionalDetr
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None, top_k: int = 100
    ):
        """
        Converts the raw output of [`ConditionalDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            top_k (`int`, *optional*, defaults to 100):
                Keep only top k bounding boxes before filtering by thresholding.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # Extract logits and bounding boxes from the model's outputs
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # Verify if target_sizes are provided and match the batch dimension of logits
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Apply sigmoid activation to logits to obtain probabilities
        prob = out_logits.sigmoid()
        # Reshape probabilities to (batch_size, num_classes)
        prob = prob.view(out_logits.shape[0], -1)
        # Determine the number of top-k predictions to consider
        k_value = min(top_k, prob.size(1))
        # Extract top-k values and their indices along the second dimension
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        # Scores correspond to the top-k values
        scores = topk_values
        # Convert top-k indexes to top-k boxes in relative [0, 1] coordinates
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        # Extract labels from top-k indexes
        labels = topk_indexes % out_logits.shape[2]
        # Convert predicted boxes from center-offset format to (x1, y1, x2, y2) format
        boxes = center_to_corners_format(out_bbox)
        # Gather top-k boxes based on top-k indexes
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Convert relative [0, 1] coordinates to absolute [0, height] coordinates if target_sizes are provided
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                # If target_sizes is a list, extract heights and widths
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                # If target_sizes is a tensor, unbind heights and widths
                img_h, img_w = target_sizes.unbind(1)
            # Stack widths and heights and scale boxes accordingly
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Filter predictions based on the score threshold and construct result dictionaries
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_semantic_segmentation with Detr->ConditionalDetr
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple[int, int]] = None):
        """
        Converts the output of [`ConditionalDetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`ConditionalDetrForSegmentation`]):
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
        # Extract class logits from model outputs [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.logits

        # Extract mask logits from model outputs [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.pred_masks

        # Remove the null class from class logits using softmax, leaving out the last dimension (background class)
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]

        # Apply sigmoid to mask logits to get probabilities [batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()

        # Compute semantic segmentation logits by combining class and mask probabilities [batch_size, num_classes, height, width]
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps if target_sizes are provided
        if target_sizes is not None:
            # Ensure that the number of target sizes matches the batch size
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            # Iterate over each image in the batch
            for idx in range(batch_size):
                # Resize logits to match target size using bilinear interpolation
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # Extract semantic segmentation map by taking the argmax along the channel dimension
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # If target_sizes are not provided, compute semantic segmentation by taking argmax along the channel dimension
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_instance_segmentation with Detr->ConditionalDetr
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
    # 从 transformers.models.conditional_detr.image_processing_conditional_detr.ConditionalDetrImageProcessor.post_process_panoptic_segmentation 复制而来
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
```