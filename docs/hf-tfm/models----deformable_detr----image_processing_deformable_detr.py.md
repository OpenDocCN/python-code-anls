# `.\models\deformable_detr\image_processing_deformable_detr.py`

```
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
"""Image processor class for Deformable DETR."""

import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

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


if is_torch_available():
    import torch
    from torch import nn


if is_vision_available():
    import PIL

if is_scipy_available():
    import scipy.special
    import scipy.stats


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# Copied from transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio
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
    # 解构输入的图片尺寸元组，分别取出高度和宽度
    height, width = image_size
    # 如果指定了最大尺寸限制
    if max_size is not None:
        # 计算原始尺寸中的最小值
        min_original_size = float(min((height, width)))
        # 计算原始尺寸中的最大值
        max_original_size = float(max((height, width)))
        # 如果根据最大原始尺寸调整后的尺寸超过了最大限制，则重新调整尺寸
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    
    # 如果高度小于等于宽度且高度等于目标尺寸，或者宽度小于等于高度且宽度等于目标尺寸，则直接返回原始高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    
    # 根据图像的宽高比例调整输出的宽高
    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    
    # 返回调整后的输出高度和宽度
    return (oh, ow)
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
    # 获取输入图像的尺寸
    image_size = get_image_size(input_image, input_data_format)
    
    # 如果输出尺寸是元组或列表，则直接返回
    if isinstance(size, (list, tuple)):
        return size
    
    # 否则根据输入图像的尺寸和输出的单一尺寸计算具有保持宽高比的输出尺寸
    return get_size_with_aspect_ratio(image_size, size, max_size)


# Copied from transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    # 如果输入数组是 numpy 数组，则返回 numpy 的 array 函数
    if isinstance(arr, np.ndarray):
        return np.array
    
    # 如果 TensorFlow 可用且输入数组是 TensorFlow 张量，则返回 TensorFlow 的 convert_to_tensor 函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf
        return tf.convert_to_tensor
    
    # 如果 PyTorch 可用且输入数组是 PyTorch 张量，则返回 PyTorch 的 tensor 函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch
        return torch.tensor
    
    # 如果 Flax 可用且输入数组是 JAX 张量，则返回 JAX 的 array 函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp
        return jnp.array
    
    # 如果无法识别输入数组的类型，则抛出 ValueError
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# Copied from transformers.models.detr.image_processing_detr.safe_squeeze
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    # 如果未指定轴，则按默认行为挤压数组
    if axis is None:
        return arr.squeeze()
    
    # 否则尝试按指定轴挤压数组，若失败则返回原数组
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# Copied from transformers.models.detr.image_processing_detr.normalize_annotation
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    # 从图像尺寸元组中获取高度和宽度
    image_height, image_width = image_size
    
    # 初始化归一化后的注释字典
    norm_annotation = {}
    # 遍历注释字典中的每个键值对
    for key, value in annotation.items():
        # 如果键是 "boxes"
        if key == "boxes":
            # 将值赋给变量 boxes
            boxes = value
            # 转换边界框格式为中心点表示，并归一化到图像尺寸
            boxes = corners_to_center_format(boxes)
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的边界框数据存入 norm_annotation 字典
            norm_annotation[key] = boxes
        else:
            # 对于其他键直接存入 norm_annotation 字典
            norm_annotation[key] = value
    # 返回归一化后的注释字典
    return norm_annotation
# 从 `transformers.models.detr.image_processing_detr.max_across_indices` 模块中复制的函数，用于返回可迭代值中每个索引的最大值列表。
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回一个可迭代值中所有索引的最大值列表。
    """
    return [max(values_i) for values_i in zip(*values)]


# 从 `transformers.models.detr.image_processing_detr.get_max_height_width` 模块中复制的函数，用于获取批次中所有图像的最大高度和宽度。
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


# 从 `transformers.models.detr.image_processing_detr.make_pixel_mask` 模块中复制的函数，用于生成图像的像素掩码。
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    生成图像的像素掩码，其中 1 表示有效像素，0 表示填充像素。

    Args:
        image (`np.ndarray`):
            要生成像素掩码的图像。
        output_size (`Tuple[int, int]`):
            掩码的输出尺寸。
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# 从 `transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask` 模块中复制的函数，用于将 COCO 多边形注释转换为掩码。
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    将 COCO 多边形注释转换为掩码。

    Args:
        segmentations (`List[List[float]]`):
            多边形列表，每个多边形由一组 x-y 坐标表示。
        height (`int`):
            掩码的高度。
        width (`int`):
            掩码的宽度。
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks
# 从transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation复制并将DETR更改为DeformableDetr
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    将COCO格式的目标转换为DeformableDetr期望的格式。
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 提取图像ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注释
    annotations = target["annotations"]
    # 过滤掉“iscrowd”属性为1的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 提取类别ID
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 为了转换为COCO API格式
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 提取边界框信息
    boxes = [obj["bbox"] for obj in annotations]
    # 处理无边界框的情况
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]  # 将(x_min, y_min, width, height)转换为(x_min, y_min, x_max, y_max)
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)   # 裁剪边界框的x坐标
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)  # 裁剪边界框的y坐标

    # 保留有效的边界框
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果注释中包含关键点信息，则提取并添加到新的目标字典中
    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码，则提取分割掩码并添加到新的目标字典中
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
        masks: 格式为`[number_masks, height, width]`的掩码，其中N是掩码的数量

    Returns:
        boxes: 格式为`[number_masks, 4]`的边界框，xyxy格式
    """
    # 如果掩码数组为空，则返回一个形状为 (0, 4) 的零数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取掩码数组的高度 h 和宽度 w
    h, w = masks.shape[-2:]

    # 创建一维数组 y 和 x，分别表示高度和宽度范围，数据类型为 np.float32
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)

    # 创建二维网格，用 y 和 x 数组作为坐标，并按照 'ij' 索引顺序
    y, x = np.meshgrid(y, x, indexing="ij")

    # 将掩码数组与 x 数组进行逐元素相乘，得到 x_mask
    x_mask = masks * np.expand_dims(x, axis=0)

    # 对 x_mask 进行重塑和最大值计算，得到 x_max
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)

    # 使用掩码创建一个掩码数组的掩码对象，并填充未掩码部分为 1e8
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))

    # 对填充后的 x_min 进行重塑和最小值计算，得到 x_min
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 将掩码数组与 y 数组进行逐元素相乘，得到 y_mask
    y_mask = masks * np.expand_dims(y, axis=0)

    # 对 y_mask 进行重塑和最大值计算，得到 y_max
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)

    # 使用掩码创建一个掩码数组的掩码对象，并填充未掩码部分为 1e8
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))

    # 对填充后的 y_min 进行重塑和最小值计算，得到 y_min
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 返回堆叠的 x_min, y_min, x_max, y_max 数组，形状为 (N, 4)，其中 N 是掩码数量
    return np.stack([x_min, y_min, x_max, y_max], 1)
# Copied from transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation with DETR->DeformableDetr
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for DeformableDetr.
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 构建注释文件的路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    # 初始化新的目标字典
    new_target = {}
    # 将图像ID转换为numpy数组形式存储在新的目标字典中
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 将图像尺寸存储在新的目标字典中
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 将原始图像尺寸存储在新的目标字典中
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    # 如果目标字典包含分段信息
    if "segments_info" in target:
        # 从注释文件中读取掩码信息并转换为numpy数组
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将RGB格式的掩码转换为类别ID格式的掩码
        masks = rgb_to_id(masks)

        # 从segments_info中提取分段信息中的ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 使用类别ID掩码创建掩码数组
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        
        # 如果需要返回掩码，则存储在新的目标字典中
        if return_masks:
            new_target["masks"] = masks
        
        # 将掩码转换为边界框格式并存储在新的目标字典中
        new_target["boxes"] = masks_to_boxes(masks)
        
        # 提取分段信息中的类别ID并存储在新的目标字典中
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        
        # 提取分段信息中的iscrowd标志并存储在新的目标字典中
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        
        # 提取分段信息中的区域面积并存储在新的目标字典中
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    # 返回处理后的新的目标字典
    return new_target


# Copied from transformers.models.detr.image_processing_detr.get_segmentation_image
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 提取输入图像的高度和宽度
    h, w = input_size
    # 提取目标图像的最终高度和宽度
    final_h, final_w = target_size

    # 对掩码执行softmax操作，以获得每个像素最可能的类别ID
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果掩码的类别ID数量为0，则创建全零矩阵
    if m_id.shape[-1] == 0:
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取最大概率类别ID，并将其重新形状为原始图像尺寸
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重复
    if deduplicate:
        # 合并具有相同类别的掩码
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将类别ID图像转换为RGB图像
    seg_img = id_to_rgb(m_id)
    # 将图像大小调整为目标尺寸并使用最近邻插值
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    # 返回分割图像
    return seg_img


# Copied from transformers.models.detr.image_processing_detr.get_mask_area
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 提取目标图像的最终高度和宽度
    final_h, final_w = target_size
    # 将分割图像转换为numpy数组，并将其形状调整为最终图像尺寸
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 将RGB格式的图像转换为类别ID格式的图像
    m_id = rgb_to_id(np_seg_img)
    # 返回类别ID图像
    # 计算每个类别的样本数，返回一个列表，列表索引对应类别编号
    area = [(m_id == i).sum() for i in range(n_classes)]
    # 返回计算出的各类别样本数列表作为结果
    return area
# 定义函数，从类别概率的对数输出中计算标签和分数
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 对类别概率进行 softmax 处理，使其变成概率分布
    probs = scipy.special.softmax(logits, axis=-1)
    # 获取每个样本中概率最高的类别标签
    labels = probs.argmax(-1, keepdims=True)
    # 根据标签取出对应的概率作为分数
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 去除多余的维度，使得 scores 和 labels 变为一维数组
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# 定义函数，处理单个样本的 Panoptic 分割输出
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
    # 根据类别概率计算标签和分数
    scores, labels = score_labels_from_class_probabilities(out_logits)
    # 筛选出有效预测结果，去除空查询和低于阈值的检测结果
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    # 从筛选后的结果中取出分数、类别和边界框
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    # 检查每个类别是否都有相应的边界框
    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    # 取出当前有效预测的掩膜，并调整大小以匹配预处理后的图像尺寸
    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # 将掩膜展平，以便后续合并同一类别的多个掩膜
    cur_masks = cur_masks.reshape(b, -1)
    # 创建一个 defaultdict，用于跟踪每个物体类别的掩膜 ids（后续将这些 ids 合并）
    stuff_equiv_classes = defaultdict(list)
    # 遍历当前类别列表，并使用枚举函数获取索引和标签
    for k, label in enumerate(cur_classes):
        # 如果当前标签对应的不是物体类别，则将索引添加到对应的“stuff”等价类别列表中
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    # 生成分割图像，传入当前掩膜、处理后的大小、目标大小、等价类别映射和去重标志
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    
    # 获取掩膜的面积，传入当前掩膜、处理后的大小以及当前类别数
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # 如果当前类别列表非空
    if cur_classes.size() > 0:
        # 过滤面积小于等于4的掩膜
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        # 只要还有被过滤的掩膜存在就继续循环
        while filtered_small.any():
            # 从当前掩膜、分数和类别中移除面积小于等于4的掩膜
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            # 重新生成分割图像，传入处理后的大小、目标大小、等价类别映射和去重标志
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            # 获取更新后的掩膜的面积
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            # 重新过滤面积小于等于4的掩膜
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果当前类别列表为空，则创建一个包含一个元素的numpy数组，元素为1，数据类型为int64
        cur_classes = np.ones((1, 1), dtype=np.int64)

    # 创建segments_info列表，每个元素是一个字典，包含id、是否物体、类别id和面积信息
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除cur_classes变量
    del cur_classes

    # 使用io.BytesIO创建一个字节流对象out
    with io.BytesIO() as out:
        # 将seg_img转换为PIL图像格式，并保存到out字节流中，格式为PNG
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 构建预测结果字典，包含PNG图像字符串和segments_info列表
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    # 返回预测结果字典
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
    # 计算目标尺寸与原始尺寸的比率
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    # 创建新的注释字典，设定大小为目标尺寸
    new_annotation = {}
    new_annotation["size"] = target_size

    # 遍历原始注释的键值对
    for key, value in annotation.items():
        # 如果键是"boxes"，则将边界框按比例缩放
        if key == "boxes":
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes
        # 如果键是"area"，则将面积按比例缩放
        elif key == "area":
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area
        # 如果键是"masks"，则按目标尺寸和指定的重采样方法调整掩码
        elif key == "masks":
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold  # 使用阈值二值化掩码
            new_annotation["masks"] = masks
        # 如果键是"size"，则直接设定大小为目标尺寸
        elif key == "size":
            new_annotation["size"] = target_size
        # 其他情况下直接复制原始注释的键值对
        else:
            new_annotation[key] = value

    # 返回调整后的新注释字典
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
    # 如果输入的掩码是 PyTorch 张量，则转换为 NumPy 数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将掩码展平为一维数组
    pixels = mask.flatten()
    # 在数组两端各添加一个零，以处理掩码边界
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到连续不同像素值的起始和结束索引，构建 RLE 编码
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    # 获取唯一的分割标识符列表，即所有不同的分割或类别标识符
    segment_ids = torch.unique(segmentation)

    # 初始化存储所有分割标识符的运行长度编码列表
    run_length_encodings = []
    # 遍历每个分割标识符
    for idx in segment_ids:
        # 创建一个二进制掩码，其中分割标识符对应的位置为1，其它位置为0
        mask = torch.where(segmentation == idx, 1, 0)
        # 将二进制掩码转换为运行长度编码（RLE）
        rle = binary_mask_to_rle(mask)
        # 将当前分割标识符的运行长度编码添加到列表中
        run_length_encodings.append(rle)

    # 返回所有分割标识符的运行长度编码列表
    return run_length_encodings
# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
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
    # 检查输入张量的第一个维度是否匹配
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    # 根据阈值和标签数筛选保留的对象
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    """
    Determine the validity of a segment based on mask labels and probabilities.

    Args:
        mask_labels (`torch.Tensor`):
            Tensor indicating mask labels.
        mask_probs (`torch.Tensor`):
            Tensor of mask probabilities.
        k (`int`):
            Class index to evaluate.
        mask_threshold (`float`, optional):
            Threshold value for binarizing masks. Default is 0.5.
        overlap_mask_area_threshold (`float`, optional):
            Threshold for determining valid segment based on area overlap. Default is 0.8.
    Returns:
        `Tuple[bool, torch.Tensor]`: A tuple indicating segment validity and the mask for the class `k`.
    """
    # 获取与类别 k 相关的掩码
    mask_k = mask_labels == k
    # 计算类别 k 的掩码区域面积
    mask_k_area = mask_k.sum()

    # 计算查询 k 中所有内容的区域面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 检查掩码是否存在
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除断开的小段
    if mask_exists:
        # 计算区域比例
        area_ratio = mask_k_area / original_area
        # 如果区域比例低于阈值，则认为掩码不存在
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
    """
    Compute segments based on mask probabilities, prediction scores, and labels.

    Args:
        mask_probs (`torch.Tensor`):
            Tensor of mask probabilities.
        pred_scores (`torch.Tensor`):
            Tensor of prediction scores.
        pred_labels (`torch.Tensor`):
            Tensor of prediction labels.
        mask_threshold (`float`, optional):
            Threshold value for binarizing masks. Default is 0.5.
        overlap_mask_area_threshold (`float`, optional):
            Threshold for determining valid segment based on area overlap. Default is 0.8.
        label_ids_to_fuse (`Optional[Set[int]]`, optional):
            Set of label IDs to fuse. Default is None.
        target_size (`Tuple[int, int]`, optional):
            Tuple specifying target size. Default is None.
    Returns:
        `torch.Tensor`: Segmentation results as a tensor of integers.
    """
    # 根据目标大小或默认大小获取高度和宽度
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 初始化分割结果
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    # 初始化段列表
    segments: List[Dict] = []

    # 如果有指定目标大小，则插值调整 mask_probs
    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    # 当前段的 ID
    current_segment_id = 0

    # 根据预测分数加权每个掩码
    mask_probs *= pred_scores.view(-1, 1, 1)
    # 确定每个像素的主要标签
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    # 初始化一个空字典，用于存储物体类别和其对应的当前段的标识符
    stuff_memory_list: Dict[str, int] = {}
    
    # 遍历预测标签的每一行
    for k in range(pred_labels.shape[0]):
        # 获取当前预测的类别标签
        pred_class = pred_labels[k].item()
        
        # 检查当前类别是否需要融合
        should_fuse = pred_class in label_ids_to_fuse
    
        # 检查当前索引 k 对应的掩码是否有效且足够大作为一个段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )
    
        # 如果存在有效掩码
        if mask_exists:
            # 如果当前预测类别已经在 stuff_memory_list 中存在
            if pred_class in stuff_memory_list:
                # 获取当前类别对应的段标识符
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 如果不存在，则增加段标识符并更新到 stuff_memory_list 中
                current_segment_id += 1
    
            # 将当前对象段添加到最终的分割地图中，使用掩码索引 mask_k
            segmentation[mask_k] = current_segment_id
            
            # 获取当前预测得分，并四舍五入保留六位小数
            segment_score = round(pred_scores[k].item(), 6)
            
            # 将当前段的信息添加到 segments 列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            
            # 如果当前类别需要融合，则更新 stuff_memory_list 中对应类别的段标识符
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id
    
    # 返回最终的分割地图和段列表
    return segmentation, segments
# 定义一个 Deformable DETR 图像处理器类，继承自 BaseImageProcessor 基类
class DeformableDetrImageProcessor(BaseImageProcessor):
    """
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

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.__init__中复制而来，初始化函数
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
        # 如果kwargs中存在"pad_and_return_pixel_mask"参数，则将do_pad设置为该参数值并将其从kwargs中删除
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 如果kwargs中存在"max_size"参数，则发出警告提示，推荐使用size字典中的"longest_edge"参数
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            # 否则将max_size设置为None或者size字典中的"longest_edge"参数，最大尺寸为1333
            max_size = None if size is None else 1333

        # 如果size为None，则将size设置为{"shortest_edge": 800, "longest_edge": 1333}字典
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # 调用get_size_dict函数，根据max_size和default_to_square参数调整size字典的内容
        size = get_size_dict(size, max_size=max_size, default_to_square=False)

        # 兼容处理，如果do_convert_annotations为None，则设置其值等于do_normalize
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        # 调用父类的初始化方法，传入kwargs中的其它参数
        super().__init__(**kwargs)
        # 设置对象的各个属性值
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
        # 设置有效的处理器键列表
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
    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.from_dict中复制而来，修改为DeformableDetr
    # 从字典中重新构建 DeformableDetrImageProcessor 对象，更新参数
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `DeformableDetrImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        # 复制输入的字典，以确保不会修改原始数据
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中有 "max_size" 参数，则更新到 image_processor_dict 中，并从 kwargs 中移除该参数
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        # 如果 kwargs 中有 "pad_and_return_pixel_mask" 参数，则更新到 image_processor_dict 中，并从 kwargs 中移除该参数
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        # 调用父类的 from_dict 方法，将更新后的 image_processor_dict 和剩余的 kwargs 传递给它
        return super().from_dict(image_processor_dict, **kwargs)

    # 从 DETR 的代码中复制，准备注释以供 DeformableDetr 使用
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
        Prepare an annotation for feeding into DeformableDetr model.
        """
        # 如果未指定格式，则使用类中定义的格式
        format = format if format is not None else self.format

        # 如果格式是 COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果未指定是否返回分割掩码，则默认为 False
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_detection_annotation 函数，准备 COCO_DETECTION 类型的注释
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果格式是 COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果未指定是否返回分割掩码，则默认为 True
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_panoptic_annotation 函数，准备 COCO_PANOPTIC 类型的注释
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 如果格式不是 COCO_DETECTION 或 COCO_PANOPTIC，则抛出异常
            raise ValueError(f"Format {format} is not supported.")
        
        # 返回处理后的目标字典
        return target

    # 从 DETR 的代码中复制，警告该方法即将被弃用，建议使用 prepare_annotation 方法代替
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用 prepare_annotation 方法来准备图像和目标
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回图像和处理后的目标
        return image, target

    # 从 DETR 的代码中复制，用于将 COCO 多边形转换为掩码的方法
    # 发出警告日志，提醒该方法即将在 v4.33 版本中移除
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用被复制的函数 convert_coco_poly_to_mask，并传递所有参数和关键字参数
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection 复制而来
    def prepare_coco_detection(self, *args, **kwargs):
        # 发出警告日志，提醒该方法即将在 v4.33 版本中移除
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用被复制的函数 prepare_coco_detection_annotation，并传递所有参数和关键字参数
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic 复制而来
    def prepare_coco_panoptic(self, *args, **kwargs):
        # 发出警告日志，提醒该方法即将在 v4.33 版本中移除
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用被复制的函数 prepare_coco_panoptic_annotation，并传递所有参数和关键字参数
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.resize 复制而来
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 定义函数 resize，用于调整图像大小
    def resize(
        image: np.ndarray,
        size: Union[int, Tuple[int, int]],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: str = "channels_last",
        input_data_format: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Union[int, Tuple[int, int]]`):
                Size to resize to. Can be an integer or a tuple of height and width.
            resample (`PILImageResampling`, optional):
                Resampling filter to use if resizing the image.
            data_format (`str`, optional):
                The channel dimension format for the output image.
            input_data_format (`str`, optional):
                The channel dimension format of the input image.
        """
        # 如果参数中包含 'max_size'，发出警告并从 kwargs 中移除该参数，将其值赋给 max_size；否则 max_size 设为 None
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 调用 get_size_dict 函数，根据参数 size 和 max_size 获得实际的调整大小结果，不默认为正方形
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 如果 size 中同时包含 'shortest_edge' 和 'longest_edge'，调用 get_resize_output_image_size 函数获取调整后的大小
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        # 如果 size 中同时包含 'height' 和 'width'，直接使用这两个值作为调整后的大小
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果 size 不符合以上格式要求，抛出 ValueError 异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 调用 resize 函数，实际执行图像调整大小的操作
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        
        # 返回调整大小后的图像
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation
    # 定义函数 resize_annotation，用于调整注释（标注）的大小以匹配调整后的图像
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
        # 调用 resize_annotation 函数，实际执行标注调整大小的操作
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    # 定义一个方法用于对图像进行重新缩放
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
        # 调用外部方法，返回重新缩放后的图像数组
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        # 调用外部方法，返回规范化后的注释字典
        return normalize_annotation(annotation, image_size=image_size)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._update_annotation_for_padded_image
    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ):
        """
        Update the annotation to reflect changes made due to image padding.

        Args:
            annotation (`Dict`):
                The annotation dictionary to update.
            input_image_size (`Tuple[int, int]`):
                The size of the original input image (height, width).
            output_image_size (`Tuple[int, int]`):
                The size of the padded output image (height, width).
            padding:
                The padding applied to the image.
            update_bboxes:
                Boolean flag indicating whether to update bounding boxes in the annotation.
        """
        # 处理由于图像填充而导致的注释更新，但未给出具体实现
        pass
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        # 创建一个新的空注释字典
        new_annotation = {}
        # 将输出图像大小添加到新注释字典中的 "size" 键
        new_annotation["size"] = output_image_size

        # 遍历现有注释字典中的每个键值对
        for key, value in annotation.items():
            # 如果键是 "masks"
            if key == "masks":
                # 获取 masks 数据
                masks = value
                # 对 masks 应用零填充，使用指定的填充模式和常量值
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                # 压缩 masks 的第一个维度，确保形状适合预期
                masks = safe_squeeze(masks, 1)
                # 将处理后的 masks 存入新注释字典中的 "masks" 键
                new_annotation["masks"] = masks
            # 如果键是 "boxes" 并且 update_bboxes 为真
            elif key == "boxes" and update_bboxes:
                # 获取 boxes 数据
                boxes = value
                # 缩放边界框坐标，以适应输出图像大小
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                # 将处理后的 boxes 存入新注释字典中的 "boxes" 键
                new_annotation["boxes"] = boxes
            # 如果键是 "size"
            elif key == "size":
                # 将输出图像大小添加到新注释字典中的 "size" 键
                new_annotation["size"] = output_image_size
            else:
                # 对于其他键，直接将其值复制到新注释字典中
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
        # 使用指定的填充模式和常量值对图像进行填充
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        # 如果存在注释数据，则更新注释以适应填充后的图像
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        # 返回填充后的图像和更新后的注释数据（如果有）
        return padded_image, annotation

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 定义类的方法 pad，用于填充图像数组，并处理相关的注释
    def pad(
        self,
        images: List[np.ndarray],  # 图像数组列表，每个元素是一个 numpy 数组
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选的注释数据，可以是单个注释或注释列表
        constant_values: Union[float, Iterable[float]] = 0,  # 填充使用的常数值，可以是单个浮点数或可迭代对象
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为 True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的数据类型，可以是字符串或张量类型
        data_format: Optional[ChannelDimension] = None,  # 图像数据的格式，可以是通道维度对象或 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的数据格式，可以是字符串或通道维度对象
        update_bboxes: bool = True,  # 是否更新边界框信息，默认为 True



    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.preprocess 复制而来的方法
    def preprocess(
        self,
        images: ImageInput,  # 图像输入，可以是单个图像或图像列表
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选的注释数据，可以是单个注释或注释列表
        return_segmentation_masks: bool = None,  # 是否返回分割掩码
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 掩码文件的路径，可以是字符串或路径对象的可选对象
        do_resize: Optional[bool] = None,  # 是否调整图像大小，可选布尔值
        size: Optional[Dict[str, int]] = None,  # 图像大小的字典，包含宽度和高度
        resample=None,  # PIL 图像重新采样方法
        do_rescale: Optional[bool] = None,  # 是否重新缩放图像，可选布尔值
        rescale_factor: Optional[Union[int, float]] = None,  # 重新缩放的因子，可以是整数或浮点数
        do_normalize: Optional[bool] = None,  # 是否归一化图像像素值，可选布尔值
        do_convert_annotations: Optional[bool] = None,  # 是否转换注释数据格式，可选布尔值
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像像素均值，可以是单个浮点数或均值列表
        image_std: Optional[Union[float, List[float]]] = None,  # 图像像素标准差，可以是单个浮点数或标准差列表
        do_pad: Optional[bool] = None,  # 是否填充图像，可选布尔值
        format: Optional[Union[str, AnnotationFormat]] = None,  # 注释数据的格式，可以是字符串或注释格式对象
        return_tensors: Optional[Union[TensorType, str]] = None,  # 返回的数据类型，可以是张量类型或字符串
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 图像数据的格式，可以是字符串或通道维度对象，默认为第一种通道维度
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的数据格式，可以是字符串或通道维度对象
        **kwargs,  # 其他参数，用于接收额外的关键字参数



    # 后处理方法 - TODO: 添加对其他框架的支持
        """
        将 [`DeformableDetrForObjectDetection`] 的原始输出转换为最终的边界框，格式为 (top_left_x, top_left_y, bottom_right_x, bottom_right_y)。仅支持 PyTorch。

        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                模型的原始输出。
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                包含批处理中每个图像的大小（高度，宽度）的张量。在评估时，这必须是原始图像大小（在任何数据增强之前）。在可视化时，这应该是数据增强后，但在填充之前的图像大小。
        Returns:
            `List[Dict]`: 一个字典列表，每个字典包含模型预测的批处理中每个图像的分数、标签和边界框。
        """
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 提取输出中的分类 logits 和边界框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出 logits 和目标大小的维度是否匹配
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标大小的每个元素是否包含批处理中每个图像的大小（h, w）
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对 logits 应用 sigmoid 函数得到概率
        prob = out_logits.sigmoid()

        # 获取每个图像中前 100 个预测的最高分和其索引
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        # 计算 topk_boxes 的索引
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        
        # 将边界框转换为 (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 格式
        boxes = center_to_corners_format(out_bbox)
        # 使用 topk_boxes 获取每个图像的 top-k 边界框
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 将相对坐标 [0, 1] 转换为绝对坐标 [0, height] 和 [0, width]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 创建结果列表，每个元素是一个字典，包含预测的分数、标签和边界框
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results
    ):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
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
        # Extract logits and predicted bounding boxes from the model's outputs
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # Check if target sizes are provided and validate their length against logits batch size
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Apply sigmoid function to logits to get probabilities and reshape them
        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)

        # Determine the number of top-k boxes to consider
        k_value = min(top_k, prob.size(1))

        # Find top-k values and their corresponding indices
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values

        # Convert top-k indexes to top-k boxes
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]

        # Convert bounding boxes from center format to corner format
        boxes = center_to_corners_format(out_bbox)

        # Gather top-k boxes from all predicted boxes
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Convert boxes from relative [0, 1] to absolute [0, height] coordinates if target sizes are provided
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Filter out boxes with scores below the threshold and construct results dictionary
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
```