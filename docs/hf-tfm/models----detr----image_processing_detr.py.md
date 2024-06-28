# `.\models\detr\image_processing_detr.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for DETR."""

import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 导入 DETR 的图像处理工具和变换函数
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
# 导入 DETR 的图像工具函数
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
# 导入通用工具函数和检测相关库
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

# 如果 PyTorch 可用，则导入 torch 和 nn 模块
if is_torch_available():
    import torch
    from torch import nn

# 如果有图像处理库可用，则导入 PIL 模块
if is_vision_available():
    import PIL

# 如果有 scipy 库可用，则导入特定的子模块
if is_scipy_available():
    import scipy.special
    import scipy.stats

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义支持的注释格式常量
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# 从原始代码库获取：https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/datasets/transforms.py#L76
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    # 如果指定了最大尺寸限制
    if max_size is not None:
        # 计算原始尺寸中较小的值
        min_original_size = float(min((height, width)))
        # 计算原始尺寸中较大的值
        max_original_size = float(max((height, width)))
        # 如果缩放后的尺寸超过了最大尺寸限制，则重新计算缩放尺寸
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    
    # 如果高度小于等于宽度且高度等于指定尺寸，或者宽度小于等于高度且宽度等于指定尺寸，则直接返回原始高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    
    # 如果宽度小于高度，则根据指定尺寸计算新的宽度和高度
    if width < height:
        ow = size
        oh = int(size * height / width)
    else:  # 否则，根据指定尺寸计算新的高度和宽度
        oh = size
        ow = int(size * width / height)
    
    # 返回计算后的新的高度和宽度
    return (oh, ow)
# 计算输出图像的大小，根据输入图像大小和期望的输出大小。如果期望的输出大小是元组或列表，则直接返回。如果期望的输出大小是整数，则保持输入图像大小的长宽比。
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Args:
        input_image (`np.ndarray`):
            要调整大小的图像。
        size (`int` or `Tuple[int, int]` or `List[int]`):
            期望的输出大小。
        max_size (`int`, *可选*):
            允许的最大输出大小。
        input_data_format (`ChannelDimension` or `str`, *可选*):
            输入图像的通道维度格式。如果未提供，则将从输入图像推断。
    """
    # 获取输入图像的大小
    image_size = get_image_size(input_image, input_data_format)
    # 如果期望的大小是元组或列表，则直接返回
    if isinstance(size, (list, tuple)):
        return size
    # 否则，根据输入图像大小和期望的大小计算保持长宽比的输出图像大小
    return get_size_with_aspect_ratio(image_size, size, max_size)


# 返回一个函数，该函数将numpy数组转换为输入数组的框架
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Args:
        arr (`np.ndarray`): 要转换的数组。
    """
    # 如果arr是numpy数组，则返回np.array函数
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果可以使用TensorFlow并且arr是TensorFlow张量，则返回tf.convert_to_tensor函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf
        return tf.convert_to_tensor
    # 如果可以使用PyTorch并且arr是PyTorch张量，则返回torch.tensor函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch
        return torch.tensor
    # 如果可以使用Flax并且arr是JAX张量，则返回jnp.array函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp
        return jnp.array
    # 如果无法转换，抛出错误
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# 如果指定的轴有维度为1，则压缩数组
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Args:
        arr (`np.ndarray`): 要压缩的数组。
        axis (`int`, *可选*): 要压缩的轴。
    """
    # 如果未指定轴，则压缩所有尺寸为1的轴
    if axis is None:
        return arr.squeeze()
    
    try:
        # 尝试压缩指定轴
        return arr.squeeze(axis=axis)
    except ValueError:
        # 如果指定轴无法压缩（尺寸不为1），则返回原数组
        return arr


# 根据图像大小对注释进行归一化处理
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    """
    Args:
        annotation (`Dict`): 要归一化的注释。
        image_size (`Tuple[int, int]`): 图像的高度和宽度。
    """
    # 获取图像的高度和宽度
    image_height, image_width = image_size
    # 初始化归一化后的注释字典
    norm_annotation = {}
    # 遍历注释的键值对
    for key, value in annotation.items():
        if key == "boxes":
            # 如果键是"boxes"，则对框的坐标进行中心坐标格式到归一化坐标的转换
            boxes = value
            boxes = corners_to_center_format(boxes)
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            norm_annotation[key] = boxes
        else:
            # 其他情况直接复制值
            norm_annotation[key] = value
    # 返回归一化后的注释字典
    return norm_annotation


# 返回可迭代值中每个索引位置的最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Args:
        values (`Iterable[Any]`): 要比较的可迭代值。
    """
    return [max(values_i) for values_i in zip(*values)]
# Copied from transformers.models.vilt.image_processing_vilt.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未提供输入数据格式，则推断第一个图像的通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据通道维度格式不同，选择不同的图像尺寸获取方式
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        # 如果输入的通道维度格式无效，则抛出异常
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.vilt.image_processing_vilt.make_pixel_mask
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
    # 获取图像的高度和宽度，根据输入的数据格式
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个全零数组作为掩码
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效像素位置（左上角至input_height、input_width范围内）置为1
    mask[:input_height, :input_width] = 1
    return mask


# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L33
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
        # 导入Pycocotools的mask模块
        from pycocotools import mask as coco_mask
    except ImportError:
        # 如果导入失败，抛出ImportError异常
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    # 遍历每个多边形（由一系列坐标表示），转换为掩码
    for polygons in segmentations:
        # 使用Pycocotools的函数将多边形转换为RLE编码格式
        rles = coco_mask.frPyObjects(polygons, height, width)
        # 解码RLE编码格式，得到掩码
        mask = coco_mask.decode(rles)
        # 如果掩码维度少于3（即没有通道维度），则添加一个维度
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # 转换为uint8类型的数组
        mask = np.asarray(mask, dtype=np.uint8)
        # 在第三维度上执行逻辑或操作，将多通道掩码转换为单通道
        mask = np.any(mask, axis=2)
        masks.append(mask)
    # 如果有掩码存在，则堆叠成一个三维数组；否则创建一个全零数组
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks


# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by DETR.
    """
    # 获取图像的高度和宽度，根据输入数据格式确定通道维度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取目标字典中的图像ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注解
    annotations = target["annotations"]
    # 过滤掉“iscrowd”为1的对象，保留未标记为crowd的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有对象的类别ID，并转换为numpy数组
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 获取所有对象的面积，转换为numpy数组
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    # 获取所有对象的iscrowd标志，如果不存在则设为0，转换为numpy数组
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取所有对象的边界框（bbox），并将其转换为numpy数组
    boxes = [obj["bbox"] for obj in annotations]
    # 防止没有边界框导致的尺寸变换问题
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 调整边界框的格式，从[x_min, y_min, width, height]变为[x_min, y_min, x_max, y_max]
    boxes[:, 2:] += boxes[:, :2]
    # 确保边界框不超出图像边界
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 根据边界框的有效性创建一个掩码
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果注解存在且第一个对象包含关键点信息
    if annotations and "keypoints" in annotations[0]:
        # 获取所有对象的关键点，并转换为numpy数组
        keypoints = [obj["keypoints"] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 根据keep掩码过滤相关的注解
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        # 重新整形关键点数组，如果没有关键点则保持原状
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码
    if return_segmentation_masks:
        # 获取所有对象的分割数据，并调用函数将其转换为掩码
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    # 返回更新后的目标字典
    return new_target
# 将给定的全景分割掩码转换为包围框

def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩码周围的边界框。

    Args:
        masks: 格式为 `[number_masks, height, width]` 的掩码数组，N 是掩码数量

    Returns:
        boxes: 格式为 `[number_masks, 4]` 的边界框数组，使用 xyxy 格式
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # 创建网格坐标，解决特定平台上的问题，详见链接
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    为 DETR 准备 COCO 全景注释。

    Args:
        image: 输入图像
        target: 包含目标信息的字典
        masks_path: 分割掩码的路径
        return_masks: 是否返回掩码
        input_data_format: 输入数据的通道维度格式

    Returns:
        new_target: 处理后的 COCO 全景注释字典
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        new_target["boxes"] = masks_to_boxes(masks)
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    """
    获取分割图像。

    Args:
        masks: 分割掩码数组
        input_size: 输入大小
        target_size: 目标大小
        stuff_equiv_classes: 材质等价类
        deduplicate: 是否去重

    Returns:
        segmentation_image: 分割图像
    """
    # 从输入大小元组中获取高度和宽度
    h, w = input_size
    # 从目标大小元组中获取最终高度和宽度
    final_h, final_w = target_size

    # 对 masks 应用 softmax 函数，转置以便按需计算
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果没有检测到任何掩模 :(
    if m_id.shape[-1] == 0:
        # 将 m_id 初始化为全零数组，数据类型为 np.int64
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取出最大概率的类别索引，并将形状转换为 (h, w)
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重
    if deduplicate:
        # 合并属于相同类别的掩模
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                # 将 m_id 中等于 eq_id 的值替换为 equiv[0]
                m_id[m_id == eq_id] = equiv[0]

    # 将 m_id 转换为 RGB 彩色分割图像
    seg_img = id_to_rgb(m_id)
    # 调整 seg_img 的大小至 (final_w, final_h)，使用最近邻插值
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    # 返回调整大小后的分割图像
    return seg_img
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
    Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions for a single sample.

    Args:
        out_logits (`torch.Tensor`):
            The logits for this sample.
        masks (`torch.Tensor`):
            The predicted segmentation masks for this sample.
        boxes (`torch.Tensor`):
            The predicted bounding boxes for this sample. The boxes are in the normalized format `(center_x, center_y,
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
    # 计算类别概率的得分和标签
    scores, labels = score_labels_from_class_probabilities(out_logits)
    
    # 保留得分高于阈值且不是背景类别的预测结果
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)
    
    # 提取保留预测结果的得分、类别和框
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])
    
    # 检查每个类别是否有对应的边界框
    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")
    
    # 提取保留预测结果的掩码，并调整大小到处理后的图像大小
    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape
    
    # 将每个掩码展平，以便后续合并同一类别的多个掩码
    cur_masks = cur_masks.reshape(b, -1)
    # 创建一个默认值为列表的 defaultdict，用于存储相似物体类别的索引列表
    stuff_equiv_classes = defaultdict(list)
    
    # 遍历当前类别列表 cur_classes 中的每个索引 k 和其对应的标签 label
    for k, label in enumerate(cur_classes):
        # 如果该标签对应的不是物体类别，则将索引 k 添加到 stuff_equiv_classes[label] 列表中
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)
    
    # 调用函数生成分割图像 seg_img，传入当前掩膜 cur_masks、处理后的大小 processed_size、目标大小 target_size、相似物体类别的索引列表 stuff_equiv_classes，并进行去重处理
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    
    # 调用函数计算当前掩膜 cur_masks 的面积 area，传入处理后的大小 processed_size 和当前分数列表 cur_scores 的长度作为参数
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))
    
    # 如果当前类别列表 cur_classes 的大小大于 0，则执行以下操作；否则将 cur_classes 设为包含一个元素的数组
    if cur_classes.size() > 0:
        # 创建布尔类型的数组 filtered_small，其元素为 True 表示对应的面积 area 小于等于 4
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        # 只要 filtered_small 中还有 True，则循环执行以下操作
        while filtered_small.any():
            # 从 cur_masks、cur_scores 和 cur_classes 中过滤掉 filtered_small 中对应为 True 的项
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            # 重新生成分割图像 seg_img，传入当前掩膜 cur_masks、处理后的大小 (h, w)、目标大小 target_size、相似物体类别的索引列表 stuff_equiv_classes，并进行去重处理
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            # 重新计算当前分割图像 seg_img 的面积 area，传入目标大小 target_size 和当前分数列表 cur_scores 的长度作为参数
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            # 更新 filtered_small，重新标记 area 中小于等于 4 的项
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果 cur_classes 的大小为 0，则将 cur_classes 设为包含一个元素的数组，元素类型为 np.int64，其值为 1
        cur_classes = np.ones((1, 1), dtype=np.int64)
    
    # 创建 segments_info 列表，其中每个元素是一个字典，包含 id、isthing、category_id 和 area 四个键值对
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除 cur_classes 变量，释放内存
    del cur_classes
    
    # 使用 io.BytesIO() 创建一个字节流对象 out
    with io.BytesIO() as out:
        # 将分割图像 seg_img 转换为 PIL.Image 对象，保存为 PNG 格式，并将结果写入到 out 字节流中
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 构建预测结果字典 predictions，包含键值对 "png_string" 和 "segments_info"
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
    
    # 返回预测结果字典 predictions
    return predictions
# 调整注释大小以适应目标大小

def resize_annotation(
    annotation: Dict[str, Any],  # 接受一个字典作为注释的输入
    orig_size: Tuple[int, int],  # 原始图像的尺寸元组，格式为 (width, height)
    target_size: Tuple[int, int],  # 目标图像的尺寸元组，格式为 (width, height)
    threshold: float = 0.5,  # 用于二值化分割掩模的阈值，默认为 0.5
    resample: PILImageResampling = PILImageResampling.NEAREST,  # 用于调整掩模大小的重采样滤波器，默认为最近邻插值
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
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))  # 计算尺寸缩放比例
    ratio_height, ratio_width = ratios

    new_annotation = {}
    new_annotation["size"] = target_size  # 将目标尺寸存入新的注释字典中

    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes  # 缩放边界框坐标并存入新的注释字典中
        elif key == "area":
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area  # 缩放区域面积并存入新的注释字典中
        elif key == "masks":
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])  # 调整掩模大小
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold  # 根据阈值二值化掩模
            new_annotation["masks"] = masks  # 存入新的注释字典中
        elif key == "size":
            new_annotation["size"] = target_size  # 如果键为"size"，更新尺寸信息
        else:
            new_annotation[key] = value  # 其他键直接复制到新的注释字典中

    return new_annotation  # 返回调整后的注释字典


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
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# TODO - (Amy) make compatible with other frameworks
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.
    ```
    # 获取唯一的分割标识符列表，表示图像分割中的不同类别或段落
    segment_ids = torch.unique(segmentation)

    # 初始化空列表，用于存储每个类别或段落的运行长度编码
    run_length_encodings = []
    
    # 遍历每个唯一的分割标识符
    for idx in segment_ids:
        # 创建一个与分割图像匹配的二进制掩码，其中与当前标识符匹配的像素值为1，否则为0
        mask = torch.where(segmentation == idx, 1, 0)
        
        # 将二进制掩码转换为运行长度编码（Run-Length Encoding，RLE）
        rle = binary_mask_to_rle(mask)
        
        # 将当前类别或段落的运行长度编码添加到结果列表中
        run_length_encodings.append(rle)

    # 返回所有类别或段落的运行长度编码列表
    return run_length_encodings
# 创建一个函数来移除低分和无对象的数据，保留符合条件的 `masks`, `scores` 和 `labels`
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
    # 检查所有输入张量的第一个维度是否相等，如果不相等则抛出 ValueError 异常
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    # 创建一个布尔索引，用于选择符合条件的对象
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    # 返回符合条件的 `masks`, `scores` 和 `labels`
    return masks[to_keep], scores[to_keep], labels[to_keep]


# 检查分割 mask 的有效性，返回是否存在符合条件的 mask 以及该类别 k 的 mask
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与类别 k 相关联的 mask
    mask_k = mask_labels == k
    # 计算类别 k 的 mask 区域面积
    mask_k_area = mask_k.sum()

    # 计算类别 k 在预测中的原始区域面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 检查是否存在类别 k 的 mask 以及原始区域面积是否大于 0
    mask_exists = mask_k_area > 0 and original_area > 0

    # 如果 mask 存在，则进一步检查区域面积比例是否大于给定阈值
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    # 返回 mask 是否存在以及类别 k 的 mask
    return mask_exists, mask_k


# 计算分割 mask 的各个段落，并返回分割结果
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    # 根据 target_size 设置高度和宽度
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 创建一个空的整数类型张量用于存储分割结果
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    # 创建一个空的字典列表，用于存储每个分割段落的信息
    segments: List[Dict] = []

    # 如果指定了 target_size，则对 mask_probs 进行双线性插值
    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    # 初始化当前分割段落的 ID
    current_segment_id = 0

    # 根据预测得分加权每个 mask
    mask_probs *= pred_scores.view(-1, 1, 1)
    # 找到每个像素位置的预测类别标签
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 记录每个类别的实例数量
    stuff_memory_list: Dict[str, int] = {}
    # 对预测标签的每个样本进行循环处理
    for k in range(pred_labels.shape[0]):
        # 获取当前样本的预测类别
        pred_class = pred_labels[k].item()
        # 判断当前类别是否需要融合
        should_fuse = pred_class in label_ids_to_fuse

        # 检查当前样本的分割掩码是否存在并且足够大以成为一个段落
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        # 如果存在有效的分割掩码
        if mask_exists:
            # 如果当前类别在stuff_memory_list中已存在，获取其对应的段落ID
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 否则，增加当前段落ID并分配给当前类别
                current_segment_id += 1

            # 将当前对象段落添加到最终的分割图中
            segmentation[mask_k] = current_segment_id
            # 获取当前样本的预测分数，并四舍五入保留6位小数
            segment_score = round(pred_scores[k].item(), 6)
            # 将当前段落信息添加到segments列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            # 如果需要融合，则更新stuff_memory_list中当前类别对应的段落ID
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    # 返回最终的分割图和segments列表作为结果
    return segmentation, segments
class DetrImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's `(height, width)` dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to True):
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
        # 如果在 `kwargs` 中有 "pad_and_return_pixel_mask"，则将 `do_pad` 设置为对应的值，并移除该参数
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 如果 `kwargs` 中有 "max_size"，发出一次警告，并将其移除；建议使用 `size['longest_edge']` 进行设置
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None if size is None else 1333

        # 如果 `size` 为 `None`，则设置默认的尺寸字典，其中包括 "shortest_edge" 和 "longest_edge" 的默认值
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # 根据给定的 `size` 和 `max_size`，获取调整后的尺寸字典，确保图像尺寸的合理性
        size = get_size_dict(size, max_size=max_size, default_to_square=False)

        # 兼容性处理：如果 `do_convert_annotations` 为 `None`，则设置为 `do_normalize` 的值
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        # 调用父类的初始化方法，传递 `kwargs` 中的参数
        super().__init__(**kwargs)
        # 设置对象的各种属性，用于图像处理流程中的参数控制和数据处理
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
        # 定义有效的处理器键列表，用于验证和访问处理器对象的属性
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
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        从字典创建图像处理器的实例，可以通过此方法更新参数，例如通过 `DetrImageProcessor.from_pretrained(checkpoint, size=600, max_size=800)` 创建图像处理器。
        """
        # 复制输入的字典，以确保不修改原始数据
        image_processor_dict = image_processor_dict.copy()
        # 如果 `kwargs` 中包含 "max_size"，则更新 `image_processor_dict` 中的 "max_size" 参数，并从 `kwargs` 中移除
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果 `kwargs` 中包含 "pad_and_return_pixel_mask"，则更新 `image_processor_dict` 中的 "pad_and_return_pixel_mask" 参数，并从 `kwargs` 中移除
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        # 调用父类的 `from_dict` 方法，传递更新后的 `image_processor_dict` 和未处理的 `kwargs`
        return super().from_dict(image_processor_dict, **kwargs)

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
        准备一个用于输入 DETR 模型的注释。
        """
        # 如果未指定格式，则使用默认格式 `self.format`
        format = format if format is not None else self.format

        # 根据注释格式调用相应的准备函数来准备注释数据
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果 `return_segmentation_masks` 未指定，则根据情况设置为 False
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 调用 `prepare_coco_detection_annotation` 函数来准备 COCO 检测格式的注释数据
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果 `return_segmentation_masks` 未指定，则根据情况设置为 True
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 调用 `prepare_coco_panoptic_annotation` 函数来准备 COCO 全景格式的注释数据
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 如果指定的格式不支持，则抛出 ValueError 异常
            raise ValueError(f"Format {format} is not supported.")
        
        # 返回处理后的注释数据
        return target

    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        """
        准备输入数据，调用 `prepare_annotation` 方法来处理目标注释。
        """
        # 发出警告，提示 `prepare` 方法将在 v4.33 版本中移除，建议使用 `prepare_annotation` 方法代替
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用 `prepare_annotation` 方法来处理目标注释
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回处理后的图像和注释数据
        return image, target

    def convert_coco_poly_to_mask(self, *args, **kwargs):
        """
        将 COCO 多边形格式的注释转换为掩码格式的方法，发出警告表示此方法将在 v4.33 版本中移除。
        """
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用 `convert_coco_poly_to_mask` 函数并返回其结果
        return convert_coco_poly_to_mask(*args, **kwargs)
    # 警告日志：方法已弃用，将在 v4.33 版本移除
    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用替代方法处理 COCO 检测注释数据集
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 警告日志：方法已弃用，将在 v4.33 版本移除
    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用替代方法处理 COCO 全景分割注释数据集
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 图像调整大小方法，根据指定尺寸和参数调整输入图像大小
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
        # 如果 `max_size` 在参数中，则发出警告日志并弹出该参数
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 获取调整后的大小字典，包括最大尺寸和默认不是正方形
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 根据给定的尺寸信息调整输出图像的大小
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果尺寸信息不完整，则引发值错误异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 使用指定参数调整图像大小并返回调整后的图像
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image
    # 调用全局函数 `resize_annotation` 来调整给定注释的大小，以匹配调整后的图像大小
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
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # TODO (Amy) - update to use `rescale_factor` instead of `scale`
    # 根据给定的 `rescale_factor` 缩放图像，更新后的尺寸为 `image = image * rescale_factor`
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

    # 根据给定的图像大小归一化注释框的坐标，从 `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]`
    # 转换为 `[center_x, center_y, width, height]` 格式，并将绝对像素值转换为相对值
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        return normalize_annotation(annotation, image_size=image_size)

    # 更新填充图像后的注释信息，根据输入和输出图像大小、填充值和是否更新边界框信息进行更新
    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ) -> Dict:
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        # 创建一个新的空字典用于存储更新后的注释信息
        new_annotation = {}
        # 将输出图像的尺寸信息添加到新注释字典中
        new_annotation["size"] = output_image_size

        # 遍历原始注释中的每个键值对
        for key, value in annotation.items():
            # 如果键是"masks"
            if key == "masks":
                # 获取masks值
                masks = value
                # 对masks进行填充操作，使用指定的填充模式和常数值
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                # 对填充后的masks进行安全压缩，移除维度为1的维度
                masks = safe_squeeze(masks, 1)
                # 将处理后的masks存入新注释字典中
                new_annotation["masks"] = masks
            # 如果键是"boxes"且需要更新边界框
            elif key == "boxes" and update_bboxes:
                # 获取boxes值
                boxes = value
                # 根据输入和输出图像的尺寸比例调整边界框的坐标值
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                # 将调整后的boxes存入新注释字典中
                new_annotation["boxes"] = boxes
            # 如果键是"size"
            elif key == "size":
                # 将输出图像的尺寸信息存入新注释字典中（这一步似乎是多余的，因为在初始化时已经添加过）
                new_annotation["size"] = output_image_size
            else:
                # 将其他键值对直接复制到新注释字典中
                new_annotation[key] = value

        # 返回更新后的注释字典
        return new_annotation

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

        # 计算在图像底部和右侧需要填充的像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充的配置元组
        padding = ((0, pad_bottom), (0, pad_right))
        # 对输入图像进行填充操作，使用指定的填充模式和常数值
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        
        # 如果提供了注释信息，则更新注释以匹配填充后的图像
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        
        # 返回填充后的图像及其对应的注释信息（如果有）
        return padded_image, annotation
    # 定义一个类方法 `pad`，用于填充图像数据。
    def pad(
        self,
        images: List[np.ndarray],  # 输入参数 `images` 是一个 NumPy 数组的列表，表示图像数据。
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选参数 `annotations`，可以是单个注解类型或注解类型的列表。
        constant_values: Union[float, Iterable[float]] = 0,  # 填充时的常数值，可以是单个浮点数或者浮点数的可迭代对象，默认为 0。
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为 True。
        return_tensors: Optional[Union[str, TensorType]] = None,  # 可选参数，指定返回的张量类型或者字符串。
        data_format: Optional[ChannelDimension] = None,  # 可选参数，指定数据格式。
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 可选参数，指定输入数据的格式。
        update_bboxes: bool = True,  # 是否更新边界框，默认为 True。
    
    # 定义一个类方法 `preprocess`，用于预处理图像数据。
    def preprocess(
        self,
        images: ImageInput,  # 输入参数 `images`，表示输入的图像数据。
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选参数 `annotations`，可以是单个注解类型或注解类型的列表。
        return_segmentation_masks: bool = None,  # 是否返回分割掩码，默认为 None。
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 可选参数 `masks_path`，指定掩码路径，可以是字符串或路径对象。
        do_resize: Optional[bool] = None,  # 是否调整大小，默认为 None。
        size: Optional[Dict[str, int]] = None,  # 可选参数 `size`，指定大小的字典。
        resample=None,  # PIL 图像重采样方法。
        do_rescale: Optional[bool] = None,  # 是否重新缩放，默认为 None。
        rescale_factor: Optional[Union[int, float]] = None,  # 可选参数 `rescale_factor`，重新缩放的因子。
        do_normalize: Optional[bool] = None,  # 是否归一化，默认为 None。
        do_convert_annotations: Optional[bool] = None,  # 是否转换注解，默认为 None。
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可以是单个浮点数或浮点数列表。
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，可以是单个浮点数或浮点数列表。
        do_pad: Optional[bool] = None,  # 是否填充，默认为 None。
        format: Optional[Union[str, AnnotationFormat]] = None,  # 注解格式，可以是字符串或注解格式对象。
        return_tensors: Optional[Union[TensorType, str]] = None,  # 返回的张量类型或字符串。
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 数据格式，默认为首通道优先。
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式。
        **kwargs,  # 其余关键字参数。
    
    # 后处理方法 - TODO: 添加对其他框架的支持
    # 受 https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258 启发的
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
        # 发出警告，提醒用户 `post_process` 方法即将被移除
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 获取输出中的分类分数和预测框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出的长度与目标尺寸长度是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标尺寸的形状是否为 (batch_size, 2)
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对分类分数进行 softmax 处理，得到概率分布，并获取最大概率的类别标签和分数
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将预测框转换为 [x0, y0, x1, y1] 的格式（左上角和右下角坐标）
        boxes = center_to_corners_format(out_bbox)
        # 将相对坐标 [0, 1] 转换为绝对坐标 [0, height]，乘以图片尺寸因子
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 组装结果，每个字典包含预测的分数、标签和框
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results
    # 定义一个方法用于后处理分割模型的输出，将输出转换为图像分割预测。仅支持 PyTorch。
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
        # 发出警告信息，提醒用户此函数即将在 Transformers v5 中删除，建议使用 `post_process_semantic_segmentation`。
        logger.warning_once(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_semantic_segmentation`.",
        )
        
        # 从模型输出中提取逻辑回归结果和预测的掩码
        out_logits, raw_masks = outputs.logits, outputs.pred_masks
        
        # 空标签的索引为输出 logits 的最后一个维度索引减一
        empty_label = out_logits.shape[-1] - 1
        
        # 存储预测结果的列表
        preds = []

        # 将输入转换为元组形式
        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        # 遍历每个样本的 logits、掩码和目标尺寸
        for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
            # 对 logits 进行 softmax 操作，获取每个预测的最大分数和对应的标签
            cur_scores, cur_labels = cur_logits.softmax(-1).max(-1)
            
            # 过滤掉空查询和分数低于阈值的检测
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            
            # 使用双线性插值将掩码调整至目标尺寸
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            
            # 将掩码转换为二进制值，根据 mask_threshold 进行阈值化
            cur_masks = (cur_masks.sigmoid() > mask_threshold) * 1

            # 将当前样本的分数、标签和掩码存储到预测字典中
            predictions = {"scores": cur_scores, "labels": cur_labels, "masks": cur_masks}
            preds.append(predictions)
        
        # 返回所有样本的预测结果列表
        return preds

    # 参考自 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218
    # 将模型输出转换为实例分割预测结果。仅支持 PyTorch。
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
        # 发出警告信息，提醒用户函数将在 Transformers 的 v5 版本中移除，请使用 `post_process_instance_segmentation`。
        logger.warning_once(
            "`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`.",
        )

        # 检查 orig_target_sizes 和 max_target_sizes 的长度是否相等，如果不相等则引发 ValueError。
        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError("Make sure to pass in as many orig_target_sizes as max_target_sizes")

        # 获取最大的高度和宽度值
        max_h, max_w = max_target_sizes.max(0)[0].tolist()

        # 压缩模型输出中的预测 masks，并进行插值操作，使其与 max_h 和 max_w 的尺寸一致
        outputs_masks = outputs.pred_masks.squeeze(2)
        outputs_masks = nn.functional.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )

        # 将 masks 转换为二进制值，根据给定的阈值进行阈值化，并移动到 CPU
        outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()

        # 遍历每个输出，调整 masks 的尺寸并保存到 results 中
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = nn.functional.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        # 返回处理后的结果列表
        return results

    # 受启发于 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L241
    # 受启发于 https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258
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
        # Extract logits and bounding boxes from model outputs
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # Check if target_sizes is provided and validate its dimension
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Compute softmax probabilities and extract scores and labels
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert bounding boxes from center format to [x0, y0, x1, y1]
        boxes = center_to_corners_format(out_bbox)

        # If target_sizes is provided, convert relative coordinates to absolute coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            # Compute scaling factors and apply to bounding boxes
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Filter predictions based on score threshold and construct results dictionary
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
    # 将模型输出转换为语义分割地图。仅支持 PyTorch。
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
        # 获取类别查询的 logits，形状为 [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.logits  
        # 获取掩码查询的 logits，形状为 [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.pred_masks  

        # 移除最后一个类别（null 类别）的 logits，使用 softmax 进行归一化
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # 使用 sigmoid 函数将掩码 logits 转换为概率，形状为 [batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()  

        # 计算语义分割 logits，形状为 (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # 调整 logits 的大小并计算语义分割地图
        if target_sizes is not None:
            # 检查目标大小的数量与 logits 的批次维度是否匹配
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                # 使用双线性插值将 logits 调整到指定大小
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 获取每个像素点的语义类别
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # 获取每个像素点的语义类别，并按批次组织成列表
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    # 受 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218 启发
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
    # 参考自 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L241
    # 定义一个方法用于处理全景分割的后处理
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
```