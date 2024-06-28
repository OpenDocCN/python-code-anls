# `.\models\deta\image_processing_deta.py`

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

import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
    PaddingMode,
    center_to_corners_format,
    corners_to_center_format,
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
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
    validate_annotations,
    validate_preprocess_arguments,
)
from ...utils import (
    is_flax_available,
    is_jax_tensor,
    is_tf_available,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from ...utils.generic import TensorType


if is_torch_available():
    import torch


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

if is_vision_available():
    import PIL


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
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    # 返回计算后的输出图像大小
    return int(round(height * size / min((height, width)))), int(round(width * size / min((height, width))))
    # 如果高度小于等于宽度且高度等于指定大小，或者宽度小于等于高度且宽度等于指定大小，则返回当前高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    # 如果宽度小于高度，则计算调整后的宽度和高度
    if width < height:
        # 新的调整后宽度为指定大小
        ow = size
        # 新的调整后高度按比例计算
        oh = int(size * height / width)
    else:
        # 否则，新的调整后高度为指定大小
        oh = size
        # 新的调整后宽度按比例计算
        ow = int(size * width / height)
    
    # 返回调整后的高度和宽度
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
    # 如果 size 是元组或列表，则直接返回 size
    if isinstance(size, (list, tuple)):
        return size
    # 否则，按照输入图像的长宽比计算输出图像的尺寸
    return get_size_with_aspect_ratio(image_size, size, max_size)


# Copied from transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    # 如果 arr 是 numpy 数组，则返回 np.array 函数
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果 TensorFlow 可用且 arr 是 TensorFlow 张量，则返回 tf.convert_to_tensor 函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果 PyTorch 可用且 arr 是 PyTorch 张量，则返回 torch.tensor 函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果 Flax 可用且 arr 是 JAX 张量，则返回 jnp.array 函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 若无法识别 arr 的类型，则引发 ValueError 异常
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# Copied from transformers.models.detr.image_processing_detr.safe_squeeze
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    # 如果未指定 axis，则对 arr 进行 squeeze 操作
    if axis is None:
        return arr.squeeze()
    
    # 尝试对指定 axis 进行 squeeze 操作，若失败则返回 arr 原样
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# Copied from transformers.models.detr.image_processing_detr.normalize_annotation
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    # 初始化规范化后的注释字典
    norm_annotation = {}
    # 遍历 annotation 字典中的每对键值对
    for key, value in annotation.items():
        # 如果当前键是 "boxes"
        if key == "boxes":
            # 将值赋给 boxes 变量
            boxes = value
            # 将边角坐标格式的 boxes 转换为中心-宽高格式
            boxes = corners_to_center_format(boxes)
            # 将 boxes 中的坐标值除以图像的宽度和高度，以实现归一化
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的 boxes 存入 norm_annotation 字典
            norm_annotation[key] = boxes
        else:
            # 对于不是 "boxes" 的键，直接将其值存入 norm_annotation 字典
            norm_annotation[key] = value
    # 返回归一化后的 annotation 字典
    return norm_annotation
# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    # 使用zip(*values)将输入的可迭代对象转置，对每个位置的元素计算最大值，返回结果列表
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未指定数据格式，使用infer_channel_dimension_format推断第一个图像的通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据数据格式分别计算最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        # 抛出异常，如果通道维度格式无效
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
    # 获取图像的高度和宽度，根据输入数据格式
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    
    # 创建一个与输出尺寸相同的零数组，将图像有效区域设为1
    mask = np.zeros(output_size, dtype=np.int64)
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
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    # 将每个多边形分割转换为掩码，合并成一个数组返回
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    
    # 如果存在掩码则堆叠成数组，否则返回一个空的数组
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks
# 从transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation中复制，用于准备COCO格式的检测注释，转换为DETA所需的格式
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    将COCO格式的目标转换为DETA期望的格式。
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取图像ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注释
    annotations = target["annotations"]
    # 过滤掉所有"isocrowd"不在对象中或者"isocrowd"等于0的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有类别ID并转换为numpy数组
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 转换为COCO API格式
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取所有边界框，并确保每个边界框均在图像内
    boxes = [obj["bbox"] for obj in annotations]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

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

    # 如果注释存在且包含关键点信息，则处理关键点
    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码，则处理分割掩码
    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


# 从transformers.models.detr.image_processing_detr.masks_to_boxes中复制，用于计算提供的全景分割掩码周围的边界框
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩码周围的边界框。

    Args:
        masks: 格式为`[number_masks, height, width]`的掩码，其中N是掩码的数量

    Returns:
        boxes: 格式为`[number_masks, 4]`的边界框，使用xyxy格式
    """
    # 如果 masks 的大小为 0，表示没有任何掩码，则返回一个形状为 (0, 4) 的全零数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取掩码的高度 h 和宽度 w
    h, w = masks.shape[-2:]
    
    # 创建一个包含从 0 到 h-1 的浮点数数组 y，以及从 0 到 w-1 的浮点数数组 x
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    
    # 使用 meshgrid 函数创建一个网格，返回的 y 和 x 是二维数组，形状为 (h, w)，使用 "ij" 索引顺序
    # 详情参见 https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    # 对 masks 应用 x 坐标，每个掩码都乘以对应的 x 坐标，并扩展维度以匹配 masks 的形状
    x_mask = masks * np.expand_dims(x, axis=0)
    
    # 对 x_mask 进行重塑成二维数组，并计算每行的最大值，得到 x 的最大值数组 x_max
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    
    # 创建一个掩码数组 x，掩盖掉所有非掩码值的元素
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    
    # 使用填充值 1e8 填充掩码外的所有元素，并对 x_min 进行重塑成二维数组，计算每行的最小值，得到 x 的最小值数组 x_min
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 对 masks 应用 y 坐标，每个掩码都乘以对应的 y 坐标，并扩展维度以匹配 masks 的形状
    y_mask = masks * np.expand_dims(y, axis=0)
    
    # 对 y_mask 进行重塑成二维数组，并计算每行的最大值，得到 y 的最大值数组 y_max
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    
    # 创建一个掩码数组 y，掩盖掉所有非掩码值的元素
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    
    # 使用填充值 1e8 填充掩码外的所有元素，并对 y_min 进行重塑成二维数组，计算每行的最小值，得到 y 的最小值数组 y_min
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 将 x_min, y_min, x_max, y_max 四个数组按列堆叠起来，形成一个形状为 (N, 4) 的数组，其中 N 是 masks 的数量
    return np.stack([x_min, y_min, x_max, y_max], 1)
# Copied from transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation with DETR->DETA
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for DETA.
    """
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 构建注释文件路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    # 设置新目标的图像ID，如果原始目标中存在'image_id'键，则使用它，否则使用'id'键
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 设置新目标的尺寸为图像的高度和宽度
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 设置新目标的原始尺寸为图像的高度和宽度
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        # 从注释路径中读取分割掩码，并将其转换为ID格式
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        # 从目标的'segments_info'中提取分割标识符ID，并根据ID创建相应的分割掩码
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        # 根据分割掩码生成边界框
        new_target["boxes"] = masks_to_boxes(masks)
        # 设置新目标的类别标签为'segments_info'中的'category_id'
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 设置新目标的'iscrowd'字段为'segments_info'中的'iscrowd'
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 设置新目标的区域面积为'segments_info'中的'area'
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


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
    # 计算原始尺寸与目标尺寸之间的比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    # 设置新注释的尺寸为目标尺寸
    new_annotation["size"] = target_size
    # 遍历注释字典中的每对键值对
    for key, value in annotation.items():
        # 如果键是"boxes"
        if key == "boxes":
            # 将值赋给变量boxes，并按比例缩放框的坐标
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            # 将缩放后的框坐标存入新的注释字典中
            new_annotation["boxes"] = scaled_boxes
        # 如果键是"area"
        elif key == "area":
            # 将值赋给变量area，并按比例缩放面积
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            # 将缩放后的面积存入新的注释字典中
            new_annotation["area"] = scaled_area
        # 如果键是"masks"
        elif key == "masks":
            # 将值赋给变量masks，并按目标大小resize每个掩码，然后进行阈值处理
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            # 将处理后的掩码存入新的注释字典中
            new_annotation["masks"] = masks
        # 如果键是"size"
        elif key == "size":
            # 直接将目标大小存入新的注释字典中
            new_annotation["size"] = target_size
        # 对于其他未处理的键，直接复制到新的注释字典中
        else:
            new_annotation[key] = value

    # 返回处理后的新注释字典
    return new_annotation
class DetaImageProcessor(BaseImageProcessor):
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
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True` will pad the images in the batch to the largest height and width in the batch.
            Padding will be applied to the bottom and right of the image with zeros.
    """
    # 定义模型输入的名称列表
    model_input_names = ["pixel_values", "pixel_mask"]

    # 初始化函数，设置数据处理的参数和选项
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
        do_convert_annotations: bool = True,
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        # 如果 kwargs 中包含 "pad_and_return_pixel_mask"，则将 do_pad 设为对应值并从 kwargs 中删除该项
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 设置图像尺寸，默认为 {"shortest_edge": 800, "longest_edge": 1333}
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # 根据 size 获取尺寸字典，确保不是默认正方形
        size = get_size_dict(size, default_to_square=False)

        # 如果 do_convert_annotations 为 None，则设为 do_normalize 的值
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        # 调用父类的初始化方法，传入可能的其他参数
        super().__init__(**kwargs)

        # 设置类的各种属性
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

    # 从 DETR 模型中复制的函数，用于准备注释信息，根据参数设置返回特定格式的注释
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
        Prepare an annotation for feeding into DETA model.
        """
        # 如果未指定格式，则使用默认格式
        format = format if format is not None else self.format

        # 根据不同的注释格式进行处理
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果未指定返回分割掩码，则默认为 False
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO 检测注释
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果未指定返回分割掩码，则默认为 True
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO 全景注释
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 抛出错误，指定的格式不支持
            raise ValueError(f"Format {format} is not supported.")

        # 返回处理后的目标注释
        return target

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 使用新的方法 `prepare_annotation` 处理注释，并返回处理后的目标注释
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return image, target

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.convert_coco_poly_to_mask
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        # 发出警告，方法 `convert_coco_poly_to_mask` 即将被移除
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用 `convert_coco_poly_to_mask` 函数，并返回结果
        return convert_coco_poly_to_mask(*args, **kwargs)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection
    def prepare_coco_detection(self, *args, **kwargs):
        # 发出警告，方法 `prepare_coco_detection` 即将被移除
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用 `prepare_coco_detection_annotation` 函数，并返回结果
        return prepare_coco_detection_annotation(*args, **kwargs)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic
    def prepare_coco_panoptic(self, *args, **kwargs):
        # 发出警告，方法 `prepare_coco_panoptic` 即将被移除
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用 `prepare_coco_panoptic_annotation` 函数，并返回结果
        return prepare_coco_panoptic_annotation(*args, **kwargs)

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
                The desired output size. Can contain keys `shortest_edge` and `longest_edge` or `height` and `width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # 获取调整后的尺寸字典，确保不默认为正方形
        size = get_size_dict(size, default_to_square=False)
        
        # 如果 size 包含 'shortest_edge' 和 'longest_edge' 键
        if "shortest_edge" in size and "longest_edge" in size:
            # 根据 'shortest_edge' 和 'longest_edge' 来计算调整后的图像尺寸
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        # 如果 size 包含 'height' 和 'width' 键
        elif "height" in size and "width" in size:
            # 直接使用 'height' 和 'width' 指定的尺寸
            size = (size["height"], size["width"])
        else:
            # 如果 size 不符合预期的键集合，抛出值错误异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 调用 resize 函数，根据指定的尺寸调整图像大小
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format
        )
        # 返回调整大小后的图像
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation
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
        # 调用外部函数 resize_annotation，将注释调整到与调整后的图像匹配的大小
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
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        return normalize_annotation(annotation, image_size=image_size)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._update_annotation_for_padded_image
    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,  # `padding` parameter for handling padding information
        update_bboxes,  # `update_bboxes` parameter to control updating bounding boxes
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        new_annotation = {}
        new_annotation["size"] = output_image_size  # 初始化新的注释字典，设置图像大小为输出图像大小

        for key, value in annotation.items():  # 遍历注释字典中的每个键值对
            if key == "masks":
                masks = value
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,  # 使用常量填充模式
                    constant_values=0,  # 填充值为0
                    input_data_format=ChannelDimension.FIRST,  # 输入数据格式为通道维度在前
                )
                masks = safe_squeeze(masks, 1)  # 压缩维度为1的维度
                new_annotation["masks"] = masks  # 更新注释字典中的masks项为填充后的masks
            elif key == "boxes" and update_bboxes:
                boxes = value
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],  # 调整框的水平位置
                        input_image_size[0] / output_image_size[0],  # 调整框的垂直位置
                        input_image_size[1] / output_image_size[1],  # 调整框的水平大小
                        input_image_size[0] / output_image_size[0],  # 调整框的垂直大小
                    ]
                )
                new_annotation["boxes"] = boxes  # 更新注释字典中的boxes项为调整后的boxes
            elif key == "size":
                new_annotation["size"] = output_image_size  # 更新注释字典中的size项为输出图像大小
            else:
                new_annotation[key] = value  # 其他键直接复制到新的注释字典中
        return new_annotation  # 返回更新后的注释字典

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
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)  # 获取输入图像的高度和宽度

        output_height, output_width = output_size  # 获取输出图像的高度和宽度

        pad_bottom = output_height - input_height  # 计算垂直方向的填充量
        pad_right = output_width - input_width  # 计算水平方向的填充量
        padding = ((0, pad_bottom), (0, pad_right))  # 设置填充的大小

        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,  # 使用常量填充模式
            constant_values=constant_values,  # 设置填充的常量值
            data_format=data_format,  # 数据格式
            input_data_format=input_data_format,  # 输入数据格式
        )

        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )  # 更新图像填充后的注释信息

        return padded_image, annotation  # 返回填充后的图像和更新后的注释信息

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 定义一个方法用于填充图像数据，支持多种参数选项
    def pad(
        self,
        images: List[np.ndarray],  # 输入参数：图像列表，每个元素是一个 NumPy 数组
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选参数：注解数据，可以是单个注解或注解列表
        constant_values: Union[float, Iterable[float]] = 0,  # 填充常数值，可以是单个浮点数或可迭代对象
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为 True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 可选参数：返回的数据类型，可以是字符串或张量类型
        data_format: Optional[ChannelDimension] = None,  # 数据格式，通道维度的定义
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，可以是字符串或通道维度对象
        update_bboxes: bool = True,  # 是否更新边界框信息，默认为 True
    ):
    
    # 定义一个方法用于预处理图像和注解数据
    def preprocess(
        self,
        images: ImageInput,  # 输入参数：图像数据，可以是单张图像或图像列表
        annotations: Optional[Union[List[Dict], List[List[Dict]]]] = None,  # 可选参数：注解数据，可以是字典列表或嵌套的字典列表
        return_segmentation_masks: bool = None,  # 是否返回分割掩码，根据情况自动设定
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 可选参数：分割掩码的路径，可以是字符串或路径对象
        do_resize: Optional[bool] = None,  # 是否调整图像大小，根据情况自动设定
        size: Optional[Dict[str, int]] = None,  # 图像大小的目标尺寸，字典形式
        resample=None,  # PIL 图像重采样方法的选项
        do_rescale: Optional[bool] = None,  # 是否进行图像重新缩放，根据情况自动设定
        rescale_factor: Optional[Union[int, float]] = None,  # 图像缩放因子，可以是整数或浮点数
        do_normalize: Optional[bool] = None,  # 是否对图像进行标准化，根据情况自动设定
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像的均值，可以是单个数值或列表
        image_std: Optional[Union[float, List[float]]] = None,  # 图像的标准差，可以是单个数值或列表
        do_convert_annotations: Optional[bool] = None,  # 是否转换注解数据的格式，根据情况自动设定
        do_pad: Optional[bool] = None,  # 是否进行图像填充，根据情况自动设定
        format: Optional[Union[str, AnnotationFormat]] = None,  # 注解数据的格式，可以是字符串或注解格式对象
        return_tensors: Optional[Union[TensorType, str]] = None,  # 返回的数据类型，可以是张量类型或字符串
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 数据格式，通道维度的定义
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，可以是字符串或通道维度对象
        **kwargs,  # 其他未明确列出的参数
    ):
    
    # 定义一个方法用于目标检测后处理
    def post_process_object_detection(
        self,
        outputs,  # 输入参数：模型输出的数据
        threshold: float = 0.5,  # 置信度阈值，默认为 0.5
        target_sizes: Union[TensorType, List[Tuple]] = None,  # 目标大小，可以是张量类型或元组列表
        nms_threshold: float = 0.7,  # 非最大抑制的阈值，默认为 0.7
```