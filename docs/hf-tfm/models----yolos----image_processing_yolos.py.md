# `.\transformers\models\yolos\image_processing_yolos.py`

```py
# 设定编码方式为 utf-8
# 版权声明
# Apache 2.0 许可
# 本文件受版权法保护，未经许可不得复制或使用
# 可以在遵守许可的前提下使用本文件
# 可以从以下网址获取许可副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面同意，根据许可分发的软件是基于“AS IS”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可协议了解有关权限和限制的具体语言。

# YOLOS 图像处理器类
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# 导入特征提取工具
from ...feature_extraction_utils import BatchFeature
# 导入图像处理工具基类
from ...image_processing_utils import BaseImageProcessor, get_size_dict
# 导入图像变换函数
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
# 导入图像工具
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
# 导入工具函数
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


# 如果可以使用 torch，导入相关模块
if is_torch_available():
    import torch
    from torch import nn

# 如果可以使用 vision，导入 PIL 模块
if is_vision_available():
    import PIL

# 如果可以使用 scipy，导入特定模块
if is_scipy_available():
    import scipy.special
    import scipy.stats

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 支持的注释格式
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)

# 从 transformers.models.detr.image_processing_detr.get_max_height_width 复制
# 获取图像批次中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 推断输入数据格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据输入数据格式计算最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)

# 根据图像尺寸、大小和最大尺寸计算具有纵横比的尺寸
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
```  
    # 根据输入图像大小和期望的输出大小计算输出图像的大小

    Args:
        image_size (`Tuple[int, int]`):
            输入图像的大小。
        size (`int`):
            期望的输出大小。
        max_size (`int`, *optional*):
            允许的最大输出大小。
    """
    # 将输入图像的高度和宽度分别赋值给变量
    height, width = image_size
    # 如果设置了最大输出大小
    if max_size is not None:
        # 计算输入图像的最小尺寸和最大尺寸
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        # 如果根据最大和最小尺寸计算得到的新尺寸超过了最大输出大小，则重新计算输出大小
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    # 如果图像宽度小于高度并且宽度不等于输出大小，则调整高度和宽度
    if width < height and width != size:
        height = int(size * height / width)
        width = size
    # 如果图像高度小于宽度并且高度不等于输出大小，则调整高度和宽度
    elif height < width and height != size:
        width = int(size * width / height)
        height = size
    # 计算宽度与16的余数
    width_mod = np.mod(width, 16)
    # 计算高度与16的余数
    height_mod = np.mod(height, 16)
    # 根据余数调整宽度和高度
    width = width - width_mod
    height = height - height_mod
    # 返回调整后的图像大小
    return (height, width)
# 从transformers.models.detr.image_processing_detr.get_resize_output_image_size复制而来
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    计算输出图像大小，给定输入图像大小和所需输出大小。如果所需输出大小是元组或列表，则返回输出图像大小。如果所需输出大小是整数，则通过保持输入图像大小的纵横比来计算输出图像大小。

    Args:
        input_image (`np.ndarray`):
            要调整大小的图像。
        size (`int` or `Tuple[int, int]` or `List[int]`):
            所需的输出大小。
        max_size (`int`, *optional*):
            允许的最大输出大小。
        input_data_format (`ChannelDimension` or `str`, *optional*):
            输入图像的通道维度格式。如果未提供，则将从输入图像推断。
    """
    # 获取输入图像的大小
    image_size = get_image_size(input_image, input_data_format)
    # 如果所需大小是列表或元组，则返回所需大小
    if isinstance(size, (list, tuple)):
        return size
    # 否则，通过保持纵横比来计算输出图像大小
    return get_size_with_aspect_ratio(image_size, size, max_size)


# 从transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn复制而来
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    返回一个函数，将numpy数组转换为输入数组的框架。

    Args:
        arr (`np.ndarray`): 要转换的数组。
    """
    # 如果输入是numpy数组，则返回np.array
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果可用tensorflow并且输入是tensorflow张量，则返回tf.convert_to_tensor
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果可用torch并且输入是torch张量，则返回torch.tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果可用flax并且输入是jax张量，则返回jnp.array
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 如果无法转换，则引发值错误
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# 从transformers.models.detr.image_processing_detr.safe_squeeze复制而来
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    挤压数组，但仅当指定的轴维度为1时。
    """
    # 如果未指定轴，则使用squeeze方法挤压数组
    if axis is None:
        return arr.squeeze()
    # 尝试挤压数组，如果出现值错误，则返回原数组
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# 从transformers.models.detr.image_processing_detr.normalize_annotation复制而来
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    norm_annotation = {}
    # 遍历 annotation 字典的键值对
    for key, value in annotation.items():
        # 如果键为"boxes"
        if key == "boxes":
            # 将值赋给 boxes
            boxes = value
            # 将 boxes 转换为中心坐标格式
            boxes = corners_to_center_format(boxes)
            # 将 boxes 归一化，即将其值除以图像的宽和高，转换为浮点型数组
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的 boxes 存入 norm_annotation 字典
            norm_annotation[key] = boxes
        # 如果键不为"boxes"
        else:
            # 将值直接存入 norm_annotation 字典
            norm_annotation[key] = value
    # 返回归一化后的 annotation 字典
    return norm_annotation
# 从 transformers.models.detr.image_processing_detr.max_across_indices 复制而来
# 返回一个可迭代值中所有索引的最大值列表
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# 从 transformers.models.detr.image_processing_detr.make_pixel_mask 复制而来
# 为图像创建像素掩码，其中 1 表示有效像素，0 表示填充
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
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    创建一个大小为 output_size 的零矩阵
    mask = np.zeros(output_size, dtype=np.int64)
    将 mask 的前 input_height 行、前 input_width 列设为1
    mask[:input_height, :input_width] = 1
    返回生成的 mask


# 从 transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask 复制而来
# 将 COCO 多边形注释转换为掩码
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
    尝试导入 pycocotools 中的 mask 模块
    如果导入失败则抛出 ImportError
    对每个多边形进行处理
        将多边形数据转换成 COCO 形式的 RLE 编码
        将 RLE 编码转换成掩码格式
        如果掩码的维度小于 3 则扩展至3维
        将掩码中为真的元素转换为 1，生成掩码数据
        沿着第3个维度进行任意运算，得到掩码
        将生成的掩码存储到 masks 列表中
    如果存在掩码数据则将其按照第0个维度进行堆叠
    否则生成一个大小为 (0, height, width) 的零矩阵
    返回存储所有掩码数据的数组


# 从 transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation 复制而来
# 将 COCO 格式中的目标转换成 DETR 期望的格式
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by DETR.
    """
    根据输入图像计算出其高度和宽度
    image_id = 从目标中获取图像 ID
    将图像 ID 转换成 int64 数据类型并存储在数组中

    从目标中获取所有 COCO 注释
    仅保留非密集注释，即“iscrowd”字段为0的注释对象

    从所有注释对象中获取类别信息并存储在 classes 数组中

    # 用于转换为 coco api
    # 将注释列表中每个对象的"area"属性提取出来，转换为浮点数类型的 numpy 数组
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    # 将注释列表中每个对象的"iscrowd"属性提取出来，如果不存在则设为0，转换为整型的 numpy 数组
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 提取每个对象的边界框信息
    boxes = [obj["bbox"] for obj in annotations]
    # 如果没有边界框则进行缩放处理
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 将边界框的(x1, y1, x2, y2)格式转换为(x1, y1, width, height)格式
    boxes[:, 2:] += boxes[:, :2]
    # 对边界框的坐标进行裁剪，确保不超出图像范围
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 根据边界框的条件(宽度大于0且高度大于0)进行筛选，得到保留的边界框索引
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]  # 保留的类别标签
    new_target["boxes"] = boxes[keep]  # 保留的边界框信息
    new_target["area"] = area[keep]  # 保留的面积信息
    new_target["iscrowd"] = iscrowd[keep]  # 保留的iscrowd信息
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)  # 图像原始尺寸信息

    # 如果注释中包含"keypoints"属性
    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将关键点列表转换为 numpy 数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 使用之前筛选出的 keep 进行关键点的筛选
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints  # 添加关键点信息到新的目标字典中

    # 如果需要返回分割掩码
    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        # 调用函数将 COCO 格式的多边形转换为掩码格式
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]  # 保留的分割掩码信息

    return new_target  # 返回新的目标字典
# 从transformers.models.detr.image_processing_detr.masks_to_boxes中复制代码
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩模周围的边界框。

    Args:
        masks: 格式为`[number_masks, height, width]`的掩模，其中N是掩模的数量

    Returns:
        boxes: 格式为`[number_masks, 4]`的边界框，使用xyxy格式
    """
    如果掩模为空，则返回形状为(0, 4)的零矩阵
    if masks.size == 0:
        return np.zeros((0, 4))

    提取掩模的高度和宽度
    h, w = masks.shape[-2:]
    创建高度范围数组
    y = np.arange(0, h, dtype=np.float32)
    创建宽度范围数组
    x = np.arange(0, w, dtype=np.float32)
    # 参考https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    对x方向掩模乘以x坐标
    x_mask = masks * np.expand_dims(x, axis=0)
    计算每个掩模的x坐标最大值
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    创建掩模的蒙版
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    用1e8填充未定义的x最小值
    x_min = x.filled(fill_value=1e8)
    计算每个掩模的x坐标最小值
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    对y方向掩模乘以y坐标
    y_mask = masks * np.expand_dims(y, axis=0)
    计算每个掩模的y坐标最大值
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    创建掩模的蒙版
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    用1e8填充未定义的y最小值
    y_min = y.filled(fill_value=1e8)
    计算每个掩模的y坐标最小值
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    返回堆叠的x最小值，y最小值，x最大值，y最大值
    return np.stack([x_min, y_min, x_max, y_max], 1)


# 从transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation中复制代码，将DETR->YOLOS
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    为YOLOS准备coco全景注释。
    """
    提取图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    获取注释的路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    新的目标字典
    new_target = {}
    如果"image_id"在目标中，则提取image_id；否则提取id
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    提取图像的大小
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    提取原始图像的大小
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 如果目标字典中包含 "segments_info" 键
    if "segments_info" in target:
        # 打开注释路径指向的图像文件，将其转换为 NumPy 数组，数据类型为 uint32
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将 RGB 格式的掩码转换为 ID 格式
        masks = rgb_to_id(masks)
    
        # 提取 "segments_info" 中每个分割信息的 ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 将掩码数组与 ID 进行逐元素比较，生成一个新的布尔掩码数组
        masks = masks == ids[:, None, None]
        # 将布尔掩码数组转换为 uint8 类型
        masks = masks.astype(np.uint8)
        # 如果需要返回掩码，将其添加到新的目标字典中
        if return_masks:
            new_target["masks"] = masks
        # 根据掩码计算目标的边界框，并添加到新的目标字典中
        new_target["boxes"] = masks_to_boxes(masks)
        # 提取每个分割信息的类别 ID，并添加到新的目标字典中
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 提取每个分割信息的是否是拥挤区域的标志，并添加到新的目标字典中
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 提取每个分割信息的面积，并添加到新的目标字典中
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )
    
    # 返回新的目标字典
    return new_target
# 从输入的掩码图像、输入尺寸、目标尺寸和 stuff_equiv_classes 中生成分割图像
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入图像的高度和宽度
    h, w = input_size
    # 获取目标尺寸的高度和宽度
    final_h, final_w = target_size

    # 对掩码图像沿通道维度应用 softmax 得到概率图
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果没有检测到任何掩码
    if m_id.shape[-1] == 0:
        # 将结果掩码设为全0
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 否则将概率图沿通道维度取argmax得到分类结果
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重
    if deduplicate:
        # 合并对应同一 stuff 类的掩码
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将分类结果转换为 RGB 图像
    seg_img = id_to_rgb(m_id)
    # 对 RGB 图像进行nearest-neighbor插值调整到目标尺寸
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# 计算分割图像中每个类别的面积
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 获取目标尺寸的高度和宽度
    final_h, final_w = target_size
    # 将分割图像转换为numpy数组并调整形状
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 将 RGB 分割图像转换为类别ID图
    m_id = rgb_to_id(np_seg_img)
    # 统计每个类别的像素个数
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area


# 根据类别概率计算标签和得分
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 对类别概率logits进行softmax归一化
    probs = scipy.special.softmax(logits, axis=-1)
    # 取概率最大的类别作为预测标签
    labels = probs.argmax(-1, keepdims=True)
    # 获取对应的概率得分
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 去掉多余的维度
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# 根据目标尺寸调整注释信息
def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
):
    # 计算宽高缩放比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    # 创建新的注释字典
    new_annotation = {}
    # 存储目标尺寸
    new_annotation["size"] = target_size
    # 遍历给定的注释字典，逐个处理其中的键值对
    for key, value in annotation.items():
        # 检查键是否为"boxes"
        if key == "boxes":
            # 如果是"boxes"，将其值赋给变量boxes
            boxes = value
            # 将边界框坐标按比例缩放，并存储在scaled_boxes中
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            # 将缩放后的边界框存储在新的注释字典中
            new_annotation["boxes"] = scaled_boxes
        # 检查键是否为"area"
        elif key == "area":
            # 如果是"area"，将其值赋给变量area
            area = value
            # 计算面积按比例缩放，并存储在scaled_area中
            scaled_area = area * (ratio_width * ratio_height)
            # 将缩放后的面积存储在新的注释字典中
            new_annotation["area"] = scaled_area
        # 检查键是否为"masks"
        elif key == "masks":
            # 如果是"masks"，将其值赋给变量masks
            masks = value[:, None]
            # 使用resize函数按目标尺寸和重采样方式对每个掩码进行缩放
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            # 将掩码数组转换为32位浮点型
            masks = masks.astype(np.float32)
            # 将掩码中大于阈值的部分设置为True，否则设置为False
            masks = masks[:, 0] > threshold
            # 将处理后的掩码存储在新的注释字典中
            new_annotation["masks"] = masks
        # 检查键是否为"size"
        elif key == "size":
            # 如果是"size"，将目标尺寸存储在新的注释字典中
            new_annotation["size"] = target_size
        else:
            # 对于其他未处理的键，直接将其值存储在新的注释字典中
            new_annotation[key] = value
    
    # 返回更新后的注释字典
    return new_annotation
# Copied from transformers.models.detr.image_processing_detr.binary_mask_to_rle
# 将给定形状为`(height, width)`的二进制掩码转换为运行长度编码（RLE）格式
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
    pixels = mask.flatten()  # 将掩码展平成一维数组
    pixels = np.concatenate([[0], pixels, [0]])  # 在掩码两侧分别添加0，以便运行长度编码计算
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 计算运行长度编码
    runs[1::2] -= runs[::2]  # 对运行长度编码进行调整
    return list(runs)  # 返回运行长度编码的列表


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
# 将给定形状为`(height, width)`的分割图转换为运行长度编码（RLE）格式
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)  # 获取分割图中的不同分割 / 类别 ID
    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)  # 获取特定分割 / 类别 ID 对应的二进制掩码
        rle = binary_mask_to_rle(mask)  # 使用binary_mask_to_rle函数获取掩码的运行长度编码
        run_length_encodings.append(rle)  # 将运行长度编码添加到列表中
    return run_length_encodings  # 返回运行长度编码的列表


# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
# 使用`object_mask_threshold`对给定的掩码进行二值化，并返回相关的值`masks`, `scores`和`labels`
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
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)  # 通过条件筛选要保留的值的索引

    return masks[to_keep], scores[to_keep], labels[to_keep]  # 返回筛选后的`masks`, `scores`和`labels`
# 检查给定类别的分割掩码是否有效
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第 k 类相关的分割掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询 k 类中所有物体的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除不连通的小片段
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

    # 根据预测分数加权每个分割掩码
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查分割掩码是否存在并且足够大以作为片段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象片段添加到最终分割地图中
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


class YolosImageProcessor(BaseImageProcessor):
    r"""
    构建一个 Detr 图像处理器
    # 定义一个方法，这个方法用于初始化一个图像预处理类
    # 方法参数:
    #   format: 注解数据的格式，可以是"coco_detection"或"coco_panoptic"之一，默认为"coco_detection"
    #   do_resize: 控制是否将图像的（高度、宽度）尺寸调整为指定的尺寸，默认为True，可以在preprocess方法的do_resize参数中覆盖
    #   size: 调整大小后图像的（高度、宽度）尺寸，默认为{"shortest_edge": 800, "longest_edge": 1333}，可以在preprocess方法的size参数中覆盖
    #   resample: 如果调整图像大小，要使用的重采样滤波器，默认为PILImageResampling.BILINEAR
    #   do_rescale: 控制是否按指定比例rescale_factor对图像进行重新缩放，默认为True，可以在preprocess方法的do_rescale参数中覆盖
    #   rescale_factor: 如果对图像进行rescale，要使用的比例因子，默认为1/255，可以在preprocess方法的rescale_factor参数中覆盖
    #   do_normalize: 控制是否对图像进行标准化，默认为True，可以在preprocess方法的do_normalize参数中覆盖
    #   image_mean: 标准化图像时使用的均值，默认为IMAGENET_DEFAULT_MEAN，可以是单个值或每个通道的值列表，可以在preprocess方法的image_mean参数中覆盖
    #   image_std: 标准化图像时使用的标准差，默认为IMAGENET_DEFAULT_STD，可以是单个值或每个通道的值列表，可以在preprocess方法的image_std参数中覆盖
    #   do_pad: 控制是否对图像进行填充以适应批处理中最大的图像并创建像素掩码，默认为True，可以在preprocess方法的do_pad参数中覆盖
    # 方法返回:
    #   无返回值
    
    # 声明model_input_names为一个包含"pixel_values"和"pixel_mask"的列表
    
    # 定义一个初始化方法，用于初始化图像预处理类
    # 方法参数:
    #   format: 注解数据的格式，可以是字符串或AnnotationFormat类型，默认为AnnotationFormat.COCO_DETECTION
    #   do_resize: 控制是否调整图像大小，默认为True
    #   size: 图像调整大小后的（高度、宽度）尺寸的字典，可为空
    #   resample: 如果调整图像大小，要使用的重采样滤波器，默认为PILImageResampling.BILINEAR
    #   do_rescale: 控制是否重新缩放图像，默认为True
    #   rescale_factor: 如果重新缩放图像，要使用的比例因子，默认为1/255
    #   do_normalize: 控制是否对图像进行标准化，默认为True
    #   image_mean: 标准化图像时使用的均值，默认为None
    #   image_std: 标准化图像时使用的标准差，默认为None
    #   do_pad: 控制是否对图像进行填充以适应批处理中最大的图像并创建像素掩码，默认为True
    #   kwargs: 其他额外参数
    # 方法返回:
    #   无返回值
    # This method is the constructor for the YolosImageProcessor class
    # It takes in various parameters related to image processing and normalization
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Union[int, Dict[str, int]]] = None,
        resample: int = Image.BILINEAR,
        do_rescale: bool = False,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = False,
        image_mean: Optional[Union[int, float, List[int], List[float]]] = None,
        image_std: Optional[Union[int, float, List[int], List[float]]] = None,
        format: str = "RGB",
        **kwargs
    ) -> None:
        # Check if the 'pad_and_return_pixel_mask' parameter is in the kwargs and store its value
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")
    
        # Check if the 'max_size' parameter is in the kwargs
        if "max_size" in kwargs:
            # Log a warning message that the 'max_size' parameter is deprecated and will be removed in the future
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`."
            )
            # Store the value of 'max_size' and remove it from the kwargs
            max_size = kwargs.pop("max_size")
        else:
            # If 'max_size' is not in the kwargs, set it to None if 'size' is None, otherwise set it to 1333
            max_size = None if size is None else 1333
    
        # Set the 'size' parameter based on the input value
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # Get the size dictionary based on the input 'size' and 'max_size' values
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
    
        # Call the super constructor with the remaining kwargs
        super().__init__(**kwargs)
    
        # Store the remaining parameters as instance variables
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
    
    # This is a class method that overrides the `from_dict` method from the base class
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `YolosImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        # Create a copy of the input image_processor_dict
        image_processor_dict = image_processor_dict.copy()
    
        # Check if the 'max_size' parameter is in the kwargs and update the image_processor_dict accordingly
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
    
        # Check if the 'pad_and_return_pixel_mask' parameter is in the kwargs and update the image_processor_dict accordingly
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
    
        # Call the super class's `from_dict` method with the updated image_processor_dict and remaining kwargs
        return super().from_dict(image_processor_dict, **kwargs)
    
    # This method is used to prepare the annotation for the image
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        This method takes an image, a target dictionary, and various optional parameters related to the annotation format,
        segmentation masks, and input data format. It processes the image and target dictionary and returns the prepared
        annotation.
        """
    # 使用类型提示注释，指明函数返回类型为Dict
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETR model.
        """
        # 根据传入的参数，确定注释的格式
        format = format if format is not None else self.format

        # 如果注释格式为COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果未指定是否返回分割掩模，则默认为False
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 准备COCO检测注释
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果注释格式为COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果未指定是否返回分割掩模，则默认为True
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 准备COCO全景注释
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        # 如果注释格式不为COCO_DETECTION或COCO_PANOPTIC，则报错
        else:
            raise ValueError(f"Format {format} is not supported.")
        # 返回处理后的注释
        return target

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare复制过来的代码
    # 准备注释，返回值不再包含图像
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        # 发出警告，此方法即将被删除
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用prepare_annotation方法处理注释
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回处理后的图像和注释
        return image, target

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.convert_coco_poly_to_mask复制过来的代码
    # 将COCO多边形转换为掩码
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        # 发出警告，此方法即将被删除
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用convert_coco_poly_to_mask方法处理COCO多边形
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection复制过来的代码
    # 准备COCO检测注释，已将DETR更改为Yolos
    def prepare_coco_detection(self, *args, **kwargs):
        # 发出警告，此方法即将被删除
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用prepare_coco_detection_annotation方法处理COCO检测注释
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic复制过来的代码
    # 准备COCO全景注释
    def prepare_coco_panoptic(self, *args, **kwargs):
        # 发出警告，此方法即将被删除
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用prepare_coco_panoptic_annotation方法处理COCO全景注释
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.resize复制过来的代码
    # 定义 resize 函数，用于将图像调整到指定大小
    def resize(
        self,
        image: np.ndarray,  # 输入图像, 以 numpy 数组的形式
        size: Dict[str, int],  # 目标大小, 可以是字典形式, 包含 'shortest_edge'、'longest_edge'、'height' 和 'width' 等关键字
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样滤波器, 默认使用双线性插值
        data_format: Optional[ChannelDimension] = None,  # 输出图像的通道维度格式, 如果未设置则使用输入图像的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的通道维度格式, 如果未提供则自动推断
        **kwargs,
    ) -> np.ndarray:
        # 处理 max_size 参数, 如果存在则将其转换为 'longest_edge' 参数
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 根据输入的 size 参数生成目标大小字典
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 根据目标大小字典计算实际的目标尺寸
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 调用 resize 函数将图像调整到目标大小, 并返回结果
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image
    
    # 定义 resize_annotation 函数, 用于调整注释的大小
    def resize_annotation(
        self,
        annotation,
        orig_size,  # 原始图像大小
        size,  # 目标大小
        resample: PILImageResampling = PILImageResampling.NEAREST,  # 重采样滤波器, 默认使用最近邻插值
    ):
    # 使用类型提示来指定函数的返回类型为字典类型
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        # 调用 resize_annotation 函数来对注释进行调整大小，匹配调整后的图像
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale 复制而来的函数
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
        # 将给定因子对图像进行重新缩放处理
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation 复制而来的函数
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format.
        """
        # 调用 normalize_annotation 函数对注释中的框进行归一化处理
        return normalize_annotation(annotation, image_size=image_size)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image 复制而来的函数
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
        # 获取输入图片的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出的高度和宽度
        output_height, output_width = output_size

        # 计算需要在图像底部和右侧填充的零的数量
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充方式的元组
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

    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def pad_images(images: List[np.ndarray], constant_values: Union[float, Iterable[float]] = 0.0,
                   return_pixel_mask: bool = True, return_tensors: Union[str, TensorType] = TensorType.UNSET,
                   data_format: Union[str, ChannelDimension] = None,
                   input_data_format: Union[str, ChannelDimension] = None) -> BatchFeature:
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
        # Get the size of the largest height and width in the batch of images
        pad_size = get_max_height_width(images, input_data_format=input_data_format)
        
        # Pad each image in the batch with zeros to match the largest height and width
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
        
        # Store the padded images in a dictionary
        data = {"pixel_values": padded_images}

        # If required, create and store pixel masks for the padded images
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return the batch feature with the data and tensor type specified
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 对输入数据进行预处理操作

    def preprocess(
        self,
        images: ImageInput,  # 输入的图像数据
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选的注释信息
        return_segmentation_masks: bool = None,  # 是否返回分割掩模
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 可选的掩模路径
        do_resize: Optional[bool] = None,  # 是否进行调整大小操作
        size: Optional[Dict[str, int]] = None,  # 大小参数
        resample=None,  # PILImageResampling  # 重新采样类型
        do_rescale: Optional[bool] = None,  # 是否进行重新缩放操作
        rescale_factor: Optional[Union[int, float]] = None,  # 重新缩放因子
        do_normalize: Optional[bool] = None,  # 是否进行标准化操作
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]] = None,  # 图像标准差
        do_pad: Optional[bool] = None,  # 是否进行填充操作
        format: Optional[Union[str, AnnotationFormat]] = None,  # 数据格式
        return_tensors: Optional[Union[TensorType, str]] = None,  # 是否返回张量
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式

        **kwargs,
        # 未指定的其他关键字参数

    # 后处理方法 - TODO: 添加对其他框架的支持
    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process 复制到 Yolos
    # 将模型的原始输出转换为最终的边界框坐标格式。只支持 PyTorch。
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`YolosForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`YolosObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 提示该函数即将被弃用，并在 v5 版本中移除，建议使用 `post_process_object_detection` 替代，使用 `threshold=0.` 来获得相同的结果
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 获取模型的 logits 和 pred_boxes
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出 logits 和目标尺寸数量是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标尺寸的形状是否正确
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对 logits 进行 softmax 操作，提取出预测的 scores 和 labels
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将边界框转换为 [x0, y0, x1, y1] 格式
        boxes = center_to_corners_format(out_bbox)
        # 将相对 [0, 1] 的坐标转换为绝对 [0, height] 的绝对坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 将 scores, labels, boxes 组合成结果字典列表
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_object_detection 复制到此处，并将 Detr->Yolos
    # 将模型对象检测的后处理功能单独定义，允许设置阈值和目标尺寸
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
        """
        将 [`YolosForObjectDetection`] 的原始输出转换为最终的边界框，格式为(top_left_x, top_left_y, bottom_right_x, bottom_right_y)。仅支持 PyTorch。

        Args:
            outputs ([`YolosObjectDetectionOutput`]):
                模型的原始输出。
            threshold (`float`, *optional*):
                保留目标检测预测结果的得分阈值。
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                形状为 `(batch_size, 2)` 的张量或包含每个图像在批处理中的目标大小`(height, width)`的元组列表。如果未设置，预测结果将不会被调整大小。
        Returns:
            `List[Dict]`: 一个字典列表，每个字典包含批处理中每个图像由模型预测的得分、标签和框。
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "确保传入与 logits 的批次维度一样多的目标大小"
                )

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 转换为 [x0, y0, x1, y1] 格式
        boxes = center_to_corners_format(out_bbox)

        # 从相对 [0, 1] 转换为绝对 [0, height] 坐标
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
```