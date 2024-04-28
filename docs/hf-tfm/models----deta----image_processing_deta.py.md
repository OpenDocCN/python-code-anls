# `.\models\deta\image_processing_deta.py`

```py
# 设置编码格式为 UTF-8
# 版权声明及许可信息
# 2022 年由 HuggingFace Inc. 团队版权所有
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于 "原样" 提供的，
# 没有任何明示或暗示的保证或条件
# 请参阅许可证了解详细信息
"""Deformable DETR 的图像处理类。"""

# 导入必要的库
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# 导入模型所需的实用程序和变换函数
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

# 如果 PyTorch 可用，则导入 PyTorch
if is_torch_available():
    import torch

# 如果 TorchVision 可用，则从 TorchVision 中导入 batched_nms 函数
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

# 如果 Vision 可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取 logger 对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 支持的注释格式
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# 从 transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio 复制
# 计算具有给定宽高比的输出图像尺寸
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    根据输入图像尺寸和期望的输出尺寸计算输出图像尺寸。

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

    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
```  
    # 如果图片的宽度小于高度
    if width < height:
        # 则新的宽度为指定大小
        ow = size
        # 新的高度按比例计算
        oh = int(size * height / width)
    else:
        # 如果图片的高度小于宽度，则新的高度为指定大小
        oh = size
        # 新的宽度按比例计算
        ow = int(size * width / height)
    # 返回计算后的宽度和高度
    return (oh, ow)
# 从transformers.models.detr.image_processing_detr.get_resize_output_image_size中复制而来
# 计算输出图像的尺寸，根据输入图像的尺寸和期望的输出尺寸。如果期望的输出尺寸是一个元组或列表，则直接返回输出图像尺寸。如果期望的输出尺寸是一个整数，则通过保持输入图像尺寸的纵横比来计算输出图像尺寸。
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
    # 如果期望的输出尺寸是一个列表或元组，则直接返回该尺寸
    if isinstance(size, (list, tuple)):
        return size
    # 否则根据期望的输出尺寸和最大允许的输出尺寸来计算输出图像尺寸
    return get_size_with_aspect_ratio(image_size, size, max_size)


# 从transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn中复制而来
# 返回一个函数，该函数将numpy数组转换为输入数组的框架
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    # 如果输入是numpy数组，则返回np.array
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果tf可用且输入为tf张量，则返回tf.convert_to_tensor
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果torch可用且输入为torch张量，则返回torch.tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果flax可用且输入为jax张量，则返回jnp.array
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 如果无法转换输入类型，则引发ValueError
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# 从transformers.models.detr.image_processing_detr.safe_squeeze中复制而来
# 只有当指定的轴维度为1时，才压缩数组
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    # 如果未指定轴，则直接进行压缩
    if axis is None:
        return arr.squeeze()
    try:
        # 尝试按指定的轴进行压缩
        return arr.squeeze(axis=axis)
    # 如果压缩失败，则返回原数组
    except ValueError:
        return arr


# 从transformers.models.detr.image_processing_detr.normalize_annotation中复制而来
# 根据图像尺寸，对注释进行标准化处理
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    # 创建一个空的标准化注释字典
    norm_annotation = {}
    # 遍历注释字典中的键值对
    for key, value in annotation.items():
        # 如果键是"boxes"
        if key == "boxes":
            # 将值赋给变量boxes
            boxes = value
            # 调用corners_to_center_format函数将边界框格式转换为中心格式
            boxes = corners_to_center_format(boxes)
            # 将边界框坐标归一化，除以图像宽度和高度，转换为浮点数数组
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的边界框存储到norm_annotation字典中的"boxes"键下
            norm_annotation[key] = boxes
        # 如果键不是"boxes"
        else:
            # 将值直接存储到norm_annotation字典中的相应键下
            norm_annotation[key] = value
    # 返回归一化后的注释字典
    return norm_annotation
# Copied from transformers.models.detr.image_processing_detr.max_across_indices
# 根据值的可迭代对象，返回每个索引上的最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]

# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
# 获取批次中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])  # 推断通道维度格式

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])  # 获取所有图像的最大高度和宽度
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])  # 获取所有图像的最大高度和宽度
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")  # 通道维度格式无效
    return (max_height, max_width)  # 返回最大高度和宽度

# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
# 为图像创建像素遮罩，其中1表示有效像素，0表示填充
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
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)  # 获取图像的高度和宽度
    mask = np.zeros(output_size, dtype=np.int64)  # 使用0填充指定形状的数组
    mask[:input_height, :input_width] = 1  # 将像素遮罩中图像部分设置为1
    return mask  # 返回像素遮罩

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
        from pycocotools import mask as coco_mask  # 导入Pycocotools模块中的mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")  # 抛出导入错误

    masks = []
    for polygons in segmentations:  # 遍历多边形列表
        rles = coco_mask.frPyObjects(polygons, height, width)  # 从多边形创建RLE编码
        mask = coco_mask.decode(rles)  # 解码RLE编码为掩码
        if len(mask.shape) < 3:  # 如果掩码维度小于3
            mask = mask[..., None]  # 添加一个轴
        mask = np.asarray(mask, dtype=np.uint8)  # 转换为NumPy数组
        mask = np.any(mask, axis=2)  # 沿第二个轴计算逻辑或
        masks.append(mask)  # 添加掩码到列表
    if masks:
        masks = np.stack(masks, axis=0)  # 沿新轴堆叠数组
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)  # 创建指定形状的零数组

    return masks  # 返回掩码
# 将目标检测的 COCO 格式数据转换为 DETA 模型需要的格式
def prepare_coco_detection_annotation(
    # 输入的图像
    image,
    # 输入的目标数据
    target,
    # 是否返回分割掩码的标志
    return_segmentation_masks: bool = False,
    # 指定输入数据格式的维度，如 "channels_first" 或 "channels_last"
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    # 获取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取目标数据中的图像 ID，并转换为 NumPy 整数数组
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取目标数据中的所有 COCO 注释
    annotations = target["annotations"]
    # 过滤掉 "iscrowd" 为 1 的注释
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有注释的类别 ID，并转换为 NumPy 整数数组
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 获取所有注释的区域面积，并转换为 NumPy 浮点数组
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    # 获取所有注释的 "iscrowd" 值，如果没有则默认为 0，并转换为 NumPy 整数数组
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取所有注释的边界框，并转换为 NumPy 浮点数组
    boxes = [obj["bbox"] for obj in annotations]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 将边界框的右下角点转换为绝对坐标
    boxes[:, 2:] += boxes[:, :2]
    # 将边界框的 x 坐标限制在图像宽度范围内
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    # 将边界框的 y 坐标限制在图像高度范围内
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 仅保留有效的边界框
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标数据字典
    new_target = {}
    # 添加图像 ID 到新的目标数据
    new_target["image_id"] = image_id
    # 添加过滤后的类别标签到新的目标数据
    new_target["class_labels"] = classes[keep]
    # 添加过滤后的边界框到新的目标数据
    new_target["boxes"] = boxes[keep]
    # 添加过滤后的区域面积到新的目标数据
    new_target["area"] = area[keep]
    # 添加过滤后的 "iscrowd" 到新的目标数据
    new_target["iscrowd"] = iscrowd[keep]
    # 添加图像原始大小到新的目标数据
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果注释包含 "keypoints"，则添加关键点到新的目标数据
    if annotations and "keypoints" in annotations[0]:
        # 获取所有注释的关键点
        keypoints = [obj["keypoints"] for obj in annotations]
        # 转换关键点为 NumPy 浮点数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 应用过滤条件，保留相关注释的关键点
        keypoints = keypoints[keep]
        # 根据关键点数量调整形状
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码，则处理分割数据
    if return_segmentation_masks:
        # 获取所有注释的分割数据
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        # 将 COCO 多边形数据转换为掩码
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        # 应用过滤条件，保留相关的掩码
        new_target["masks"] = masks[keep]

    # 返回新的目标数据
    return new_target


# 根据给定的分割掩码计算其包围盒
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算给定的泛白分割掩码的边界框。

    参数:
        masks: 分割掩码，格式为 `[number_masks, height, width]`，其中 N 是掩码的数量。

    返回:
        boxes: 边界框，格式为 `[number_masks, 4]`，xyxy 格式。
    """
    # 如果masks的大小为0，则返回一个全0数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取masks的高度和宽度
    h, w = masks.shape[-2:]

    # 生成从0到h的浮点数数组y，生成从0到w的浮点数数组x
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)

    # 生成网格坐标点的数组，使用的是"ij"索引顺序
    y, x = np.meshgrid(y, x, indexing="ij")

    # 计算x方向的最小和最大值
    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 计算y方向的最小和最大值
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 将计算出的x和y方向的最小最大值堆叠成一个数组返回
    return np.stack([x_min, y_min, x_max, y_max], 1)
# 从transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation复制以DETR->DETA
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    为DETA准备一个coco panoptic注释。
    """
    # 获取输入图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 拼接注解路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    # 将image_id添加到新目标中
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 添加图像的尺寸到新目标中
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        # 读取mask图像并转换为uint32类型
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        # 提取每个分割信息中的id并将mask二值化
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        # 将mask转换为边界框
        new_target["boxes"] = masks_to_boxes(masks)
        # 提取每个分割信息中的类别id
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 提取每个分割信息中的是否为群体标志
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 提取每个分割信息中的面积
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


# 从transformers.models.detr.image_processing_detr.resize_annotation复制
def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
):
    """
    将注解调整为目标大小。

    Args:
        annotation (`Dict[str, Any]`):
            注解字典。
        orig_size (`Tuple[int, int]`):
            输入图像的原始大小。
        target_size (`Tuple[int, int]`):
            图像的目标大小，由预处理的`resize`步骤返回。
        threshold (`float`, *optional*, 默认为0.5):
            用于二值化分割mask的阈值。
        resample (`PILImageResampling`, 默认为`PILImageResampling.NEAREST`):
            在调整mask大小时使用的重采样滤波器。
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    # 将大小属性设置为目标大小
    new_annotation["size"] = target_size
    # 遍历注释项字典的键和值
    for key, value in annotation.items():
        # 如果键为"boxes"，则进行相应处理
        if key == "boxes":
            # 将值赋给变量boxes
            boxes = value
            # 对边界框进行缩放，乘以宽高比例
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            # 更新缩放后的边界框到新注释中
            new_annotation["boxes"] = scaled_boxes
        # 如果键为"area"，则进行相应处理
        elif key == "area":
            # 将值赋给变量area
            area = value
            # 根据宽高比例对面积进行缩放
            scaled_area = area * (ratio_width * ratio_height)
            # 更新缩放后的面积到新注释中
            new_annotation["area"] = scaled_area
        # 如果键为"masks"，则进行相应处理
        elif key == "masks":
            # 将值赋给变量masks，并增加一个维度
            masks = value[:, None]
            # 对每个掩码进行尺寸调整，使其适应目标尺寸
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            # 转换掩码数据类型为浮点类型
            masks = masks.astype(np.float32)
            # 将掩码值大于阈值的像素设为True，其余设为False
            masks = masks[:, 0] > threshold
            # 更新处理后的掩码到新注释中
            new_annotation["masks"] = masks
        # 如果键为"size"，则将目标尺寸更新到新注释中
        elif key == "size":
            new_annotation["size"] = target_size
        # 对于其他键直接更新到新注释中
        else:
            new_annotation[key] = value

    # 返回处理后的新注释字典
    return new_annotation
class DetaImageProcessor(BaseImageProcessor):
    r"""
    Deformable DETR 图像处理器。

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            注释：注释可选的参数，指定注释的默认值为 "coco_detection"。
            数据格式的注释。可以是 "coco_detection" 或 "coco_panoptic" 之一。
        do_resize (`bool`, *optional*, defaults to `True`):
            注释：注释可选的参数，指定注释的默认值为 `True`。
            是否调整图像的高度和宽度至指定的 `size` 尺寸。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            注释：注释可选的参数，指定注释的默认值为 `{"shortest_edge": 800, "longest_edge": 1333}`。
            设置图像调整大小后的尺寸（高度，宽度）。可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            注释：注释可选的参数，指定注释的默认值为 `PILImageResampling.BILINEAR`。
            在调整图像大小时使用的重采样滤镜。
        do_rescale (`bool`, *optional*, defaults to `True`):
            注释：注释可选的参数，指定注释的默认值为 `True`。
            是否按照指定的比例因子 `rescale_factor` 进行图像重新缩放。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            注释：注释可选的参数，指定注释的默认值为 `1/255`。
            图像重新缩放的比例因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize:
            注释：注释可选的参数。
            是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            注释：注释可选的参数，指定注释的默认值为 `IMAGENET_DEFAULT_MEAN`。
            归一化图像时使用的均值。可以是一个值或值的列表，每个通道一个值。可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, def
    # 实例化方法，初始化数据处理类的参数
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,  # 设置默认的标注格式为COCO_DETECTION
        do_resize: bool = True,  # 是否进行调整大小，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，默认为空
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重新采样方法，默认为BILINEAR
        do_rescale: bool = True,  # 是否重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Union[float, List[float]] = None,  # 图像均值，默认为空
        image_std: Union[float, List[float]] = None,  # 图像标准差，默认为空
        do_pad: bool = True,  # 是否进行填充，默认为True
        **kwargs,  # 其他参数
    ) -> None:  # 返回值为空
        if "pad_and_return_pixel_mask" in kwargs:  # 检测是否存在关键字参数"pad_and_return_pixel_mask"
            do_pad = kwargs.pop("pad_and_return_pixel_mask")  # 如果存在，则设置do_pad为该参数的值

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}  # 如果size不为空，则使用该值，否则使用默认大小字典
        size = get_size_dict(size, default_to_square=False)  # 获取处理后的图像大小字典

        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.format = format  # 设置数据格式
        self.do_resize = do_resize  # 设置是否进行调整大小
        self.size = size  # 设置图像大小
        self.resample = resample  # 设置重新采样方法
        self.do_rescale = do_rescale  # 设置是否重新缩放
        self.rescale_factor = rescale_factor  # 设置重新缩放因子
        self.do_normalize = do_normalize  # 设置是否进行归一化
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN  # 如果图像均值不为空，则使用该值，否则使用默认值
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD  # 如果图像标准差不为空，则使用该值，否则使用默认值
        self.do_pad = do_pad  # 设置是否进行填充

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_annotation复制并修改为DETA
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,  # 标注格式，默认为空
        return_segmentation_masks: bool = None,  # 返回分割掩模的标志，默认为空
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 掩模路径，默认为空
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为空
    ) -> Dict:  # 返回类型为字典
        """
        Prepare an annotation for feeding into DETA model.
        """
        format = format if format is not None else self.format  # 如果格式不为空，则使用该值，否则使用默认格式

        if format == AnnotationFormat.COCO_DETECTION:  # 如果格式为COCO_DETECTION
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks  # 如果返回分割掩模的标志为空，则默认为False
            target = prepare_coco_detection_annotation(  # 准备COCO检测注释
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        elif format == AnnotationFormat.COCO_PANOPTIC:  # 如果格式为COCO_PANOPTIC
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks  # 如果返回分割掩模的标志为空，则默认为True
            target = prepare_coco_panoptic_annotation(  # 准备COCO全景分割注释
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:  # 如果格式不是COCO_DETECTION或COCO_PANOPTIC，则引发值错误
            raise ValueError(f"Format {format} is not supported.")
        return target  # 返回准备好的标注
    # 警告：`prepare`方法已经废弃，将在 v4.33 版本中移除。请使用`prepare_annotation`代替。
    # 注意：`prepare_annotation`方法不再返回图像。
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 使用`prepare_annotation`方法准备目标注释
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回图像和目标注释
        return image, target

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.convert_coco_poly_to_mask`复制而来
    # 警告：`convert_coco_poly_to_mask`方法已经废弃，将在 v4.33 版本中移除。
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用`convert_coco_poly_to_mask`函数
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection`复制而来
    # 警告：`prepare_coco_detection`方法已经废弃，将在 v4.33 版本中移除。
    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用`prepare_coco_detection_annotation`函数
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从`transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic`复制而来
    # 警告：`prepare_coco_panoptic`方法已经废弃，将在 v4.33 版本中移除。
    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用`prepare_coco_panoptic_annotation`函数
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 调整图像大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```py  
    ) -> np.ndarray:
        """
        调整图像大小到给定尺寸。尺寸可以是`min_size`（标量）或`(height, width)`元组。如果尺寸是整数，则图像的较小边将匹配到这个数值。

        Args:
            image (`np.ndarray`):
                需要调整大小的图像。
            size (`Dict[str, int]`):
                所需的输出尺寸。可以包含`shortest_edge`和`longest_edge`键，或`height`和`width`键。
            resample (`PILImageResampling`, *可选*, 默认为`PILImageResampling.BILINEAR`):
                调整图像时使用的重采样滤镜。
            data_format (`ChannelDimension`, *可选*):
                输出图像的通道维度格式。如果未设置，则使用输入图像的通道维度格式。
            input_data_format (`ChannelDimension`或`str`, *可选*):
                输入图像的通道维度格式。如果未提供，则将从输入图像推断。

        Returns:
            调整大小后的图像，类型为`np.ndarray`。
        """
        # 将 size 转换为标准格式字典
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            # 如果 size 包含 'shortest_edge' 和 'longest_edge' 键，调用函数计算输出图像的尺寸
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            # 如果 size 包含 'height' 和 'width' 键，直接使用这两个值作为尺寸
            size = (size["height"], size["width"])
        else:
            # 如果 size 既不包含 'height' 和 'width' 键，也不包含 'shortest_edge' 和 'longest_edge' 键，则抛出异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # 调整图像大小
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format
        )
        # 返回调整大小后的图像
        return image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation复制过来的
    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> Dict:
        """
        调整标注以匹配调整大小后的图像。如果 size 是整数，则掩码的较小边将匹配到这个数值。
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale复制过来的
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个函数，将图像按给定因子重新缩放。image = image * rescale_factor.
    def rescale_image(image: np.ndarray, rescale_factor: float, data_format: Optional[str] = None, input_data_format: Optional[str] = None) -> np.ndarray:
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
        # 返回对图像应用给定因子重新缩放的结果
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation复制而来
    # 定义一个函数，将注释中的框从`[top_left_x, top_left_y, bottom_right_x, bottom_right_y]`格式规范化为`[center_x, center_y, width, height]`格式
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format.
        """
        # 返回对注释中的框进行规范化后的结果
        return normalize_annotation(annotation, image_size=image_size)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image复制而来
    # 定义一个函数，用零填充图像到给定的尺寸
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
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        # 计算需要填充的底部和右侧像素
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        # 使用常量值填充图像
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.pad复制而来
    def pad(
        self,
        images: List[np.ndarray],  # 接受一个包含 np.ndarray 类型元素的列表作为输入 images
        constant_values: Union[float, Iterable[float]] = 0,  # 设置 padding 时要填充的常量值，默认为 0
        return_pixel_mask: bool = True,  # 是否返回像素掩码，默认为 True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，可以是 'tf'、'pt'、'np' 或 'jax' 等
        data_format: Optional[ChannelDimension] = None,  # 图像的通道维度格式，如果未提供，默认与输入图像相同
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道维度格式，如果未提供，将被推断
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
        pad_size = get_max_height_width(images, input_data_format=input_data_format)  # 获取输入图像中最大的高度和宽度

        padded_images = [  # 对每个图像进行填充操作
            self._pad_image(
                image,
                pad_size,
                constant_values=constant_values,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in images
        ]
        data = {"pixel_values": padded_images}  # 构建包含填充后图像像素值的数据字典

        if return_pixel_mask:  # 如果设置了返回像素掩码
            masks = [  # 对每个图像生成像素掩码
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks  # 将像素掩码添加到数据字典中

        return BatchFeature(data=data, tensor_type=return_tensors)  # 返回填充后的数据和指定类型的张量
    # 预处理函数，用于对输入数据进行预处理
    def preprocess(
        self,
        # 输入图像数据
        images: ImageInput,
        # 输入图像的标注信息，可选
        annotations: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        # 是否返回分割掩模
        return_segmentation_masks: bool = None,
        # 存储掩模路径，可选
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        # 是否调整大小
        do_resize: Optional[bool] = None,
        # 调整大小的目标尺寸
        size: Optional[Dict[str, int]] = None,
        # 采样方式
        resample=None,  # PILImageResampling
        # 是否重新缩放
        do_rescale: Optional[bool] = None,
        # 重新缩放的因子
        rescale_factor: Optional[Union[int, float]] = None,
        # 是否归一化
        do_normalize: Optional[bool] = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否填充
        do_pad: Optional[bool] = None,
        # 标注信息的格式
        format: Optional[Union[str, AnnotationFormat]] = None,
        # 是否返回张量
        return_tensors: Optional[Union[TensorType, str]] = None,
        # 数据格式
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        # 输入数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
    # 对目标检测结果进行后处理
    def post_process_object_detection(
        self,
        # 模型输出
        outputs,
        # 置信度阈值
        threshold: float = 0.5,
        # 目标尺寸
        target_sizes: Union[TensorType, List[Tuple]] = None,
        # NMS 阈值
        nms_threshold: float = 0.7,
```