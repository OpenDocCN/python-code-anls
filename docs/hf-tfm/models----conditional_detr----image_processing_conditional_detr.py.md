# `.\models\conditional_detr\image_processing_conditional_detr.py`

```py
# 定义了一个名为"get_size_with_aspect_ratio"的函数，用于计算输出图像大小
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            输入图像的尺寸，一个二元元组，包含图像的高度和宽度。
        size (`int`):
            所需的输出尺寸。
        max_size (`int`, *optional*):
            允许的最大输出尺寸。

    在给定输入图像大小和所需的输出大小的情况下，计算输出图像的尺寸。
    如果提供了最大尺寸参数，将根据输入图像的大小和最大尺寸调整输出尺寸。
    如果输出尺寸经过调整后大于最大尺寸，则使用最大尺寸进行限制。
    如果输出尺寸合理，则保持不变。
    """
    # 获取输入图像的高度和宽度
    height, width = image_size
    # 如果提供了最大尺寸参数
    if max_size is not None:
        # 计算输入图像的最小和最大尺寸
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        # 如果根据最大尺寸调整输出尺寸后大于最大尺寸
        if max_original_size / min_original_size * size > max_size:
            # 根据最大尺寸和输入图像的最小和最大尺寸调整输出尺寸
            size = int(round(max_size * min_original_size / max_original_size))
    # 检查图片宽高是否与指定尺寸相同，若宽度小于等于高度且宽度等于指定尺寸，或者高度小于等于宽度且高度等于指定尺寸，则返回高度和宽度
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    # 如果宽度小于高度，则根据宽度计算新的宽度和高度，使得宽度等于指定尺寸
    if width < height:
        ow = size
        oh = int(size * height / width)
    # 如果高度小于等于宽度，则根据高度计算新的宽度和高度，使得高度等于指定尺寸
    else:
        oh = size
        ow = int(size * width / height)
    return (oh, ow)
# 从transformers.models.detr.image_processing_detr.get_resize_output_image_size复制而来
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    根据输入图像大小和期望的输出大小计算输出图像大小。如果期望的输出大小是一个元组或列表，则直接返回。如果期望的输出大小是一个整数，则按照输入图像大小的宽高比进行计算。

    Args:
        input_image (`np.ndarray`):
            要调整大小的图像。
        size (`int` or `Tuple[int, int]` or `List[int]`):
            期望的输出大小。
        max_size (`int`, *optional*):
            允许的最大输出大小。
        input_data_format (`ChannelDimension` or `str`, *optional*):
            输入图像的通道维度格式。如果未提供，则将从输入图像中推断。
    """
    # 获取输入图像的大小
    image_size = get_image_size(input_image, input_data_format)
    # 如果期望的大小是一个元组或列表，则直接返回
    if isinstance(size, (list, tuple)):
        return size
    # 否则，根据输入图像大小的宽高比计算输出图像大小
    return get_size_with_aspect_ratio(image_size, size, max_size)


# 从transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn复制而来
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    返回一个将numpy数组转换为输入数组框架的函数。

    Args:
        arr (`np.ndarray`): 要转换的数组。
    """
    # 如果输入是一个numpy数组，则返回np.array函数
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果可以使用TensorFlow并且输入是一个TensorFlow张量，则返回tf.convert_to_tensor函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果可以使用PyTorch并且输入是一个PyTorch张量，则返回torch.tensor函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果可以使用Flax并且输入是一个JAX张量，则返回jnp.array函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 否则，引发异常
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# 从transformers.models.detr.image_processing_detr.safe_squeeze复制而来
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    对数组进行挤压，但仅当指定的轴维度为1时。
    """
    # 如果没有指定轴，则对数组进行挤压
    if axis is None:
        return arr.squeeze()
    # 否则，尝试对指定轴进行挤压
    try:
        return arr.squeeze(axis=axis)
    # 如果指定轴维度不为1，则返回原数组
    except ValueError:
        return arr


# 从transformers.models.detr.image_processing_detr.normalize_annotation复制而来
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    # 初始化归一化后的注释字典
    norm_annotation = {}
    # 遍历注释中的键值对
    for key, value in annotation.items():
        # 如果键是"boxes"，则将值赋给boxes变量
        if key == "boxes":
            boxes = value
            # 将boxes的坐标格式转换为中心点格式
            boxes = corners_to_center_format(boxes)
            # 对boxes进行归一化，将坐标值除以图像的宽和高，并转换为浮点数类型
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的boxes存入norm_annotation字典中
            norm_annotation[key] = boxes
        # 如果键不是"boxes"，直接将值存入norm_annotation字典中
        else:
            norm_annotation[key] = value
    # 返回归一化后的注释信息
    return norm_annotation
# 从transformers.models.detr.image_processing_detr.max_across_indices复制而来
# 返回一个可迭代值中所有索引上的最大值
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回一个可迭代值中所有索引上的最大值。
    """
    # 对于每个值的所有索引，返回最大值组成的列表
    return [max(values_i) for values_i in zip(*values)]


# 从transformers.models.detr.image_processing_detr.get_max_height_width复制而来
# 获取批次中所有图像的最大高度和宽度
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    获取批次中所有图像的最大高度和宽度。
    """
    # 如果未指定输入数据格式，则推断出通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 根据通道维度格式确定如何获取最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# 从transformers.models.detr.image_processing_detr.make_pixel_mask复制而来
# 为图像创建像素掩码，其中1表示有效像素，0表示填充。
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    为图像创建像素掩码，其中1表示有效像素，0表示填充。

    Args:
        image (`np.ndarray`):
            要创建像素掩码的图像。
        output_size (`Tuple[int, int]`):
            掩码的输出大小。
    """
    # 获取图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建全零掩码，大小与输出大小相同
    mask = np.zeros(output_size, dtype=np.int64)
    # 将图像中的有效部分填充为1
    mask[:input_height, :input_width] = 1
    return mask


# 从transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask复制而来
# 将COCO多边形标注转换为掩码
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    将COCO多边形标注转换为掩码。

    Args:
        segmentations (`List[List[float]]`):
            多边形列表，每个多边形由x-y坐标列表表示。
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
    # 对于每个多边形，将其转换为掩码
    for polygons in segmentations:
        # 将多边形列表转换为RLE编码
        rles = coco_mask.frPyObjects(polygons, height, width)
        # 解码RLE编码以获取掩码
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # 将掩码转换为二进制掩码
        mask = np.asarray(mask, dtype=np.uint8)
        # 合并掩码，获取是否存在有效像素
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        # 如果存在掩码，则堆叠它们
        masks = np.stack(masks, axis=0)
    else:
        # 如果没有掩码，则创建全零数组
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks
# 从DETR模型的图像处理模块复制prepare_coco_detection_annotation函数，将其适配为ConditionalDetr模型的格式
def prepare_coco_detection_annotation(
    image,  # 输入图像
    target,  # COCO格式的目标标注
    return_segmentation_masks: bool = False,  # 是否返回分割掩码，默认为False
    input_data_format: Optional[Union[ChannelDimension, str]] = None,  # 输入数据的格式，默认为None
):
    """
    将COCO格式的目标标注转换为ConditionalDetr所需的格式。
    """
    # 获取输入图像的高度和宽度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    # 获取图像的ID
    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO标注
    annotations = target["annotations"]
    # 过滤掉"iscrowd"键存在且值为1的标注（表示是一群对象）
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有目标的类别ID
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 用于COCO API转换的变量
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 获取所有目标的边界框
    boxes = [obj["bbox"] for obj in annotations]
    # 防止没有边界框通过调整大小
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 将边界框的坐标转换为(x_min, y_min, x_max, y_max)格式
    boxes[:, 2:] += boxes[:, :2]
    # 将边界框的坐标限制在图像范围内
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 保留有效的边界框（宽度和高度大于0）
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果标注包含关键点信息
    if annotations and "keypoints" in annotations[0]:
        # 获取所有关键点
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将筛选后的关键点列表转换为numpy数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 根据保留的掩码筛选相关的注释
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩码
    if return_segmentation_masks:
        # 获取所有分割掩码
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        # 将COCO格式的多边形分割掩码转换为掩码图像
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    # 返回新的目标字典
    return new_target


# 从DETR模型的图像处理模块复制masks_to_boxes函数
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    计算提供的全景分割掩码周围的边界框。

    参数：
        masks：格式为`[number_masks, height, width]`的掩码，其中N是掩码的数量

    返回：
        boxes：格式为`[number_masks, 4]`的边界框，以xyxy格式表示
    """
    # 如果传入的masks为空，则返回一个0行4列的数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取masks的高度和宽度
    h, w = masks.shape[-2:]

    # 创建一个包含0到h-1的一维浮点型数组
    y = np.arange(0, h, dtype=np.float32)
    # 创建一个包含0到w-1的一维浮点型数组
    x = np.arange(0, w, dtype=np.float32)

    # 根据y和x创建一个二维网格，索引方式采用"ij"
    # 参考：https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    # 将masks与扩展的x数组相乘，得到x_mask
    x_mask = masks * np.expand_dims(x, axis=0)
    # 对x_mask进行形状变换，转换为二维数组，并计算每行的最大值
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)

    # 创建一个x的掩码数组，未掩码部分填充为False
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    # 将未掩码部分填充为1e8，并对数组进行形状变换，转换为二维数组，并计算每行的最小值
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 将masks与扩展的y数组相乘，得到y_mask
    y_mask = masks * np.expand_dims(y, axis=0)
    # 对y_mask进行形状变换，转换为二维数组，并计算每行的最大值
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)

    # 创建一个y的掩码数组，未掩码部分填充为False
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    # 将未掩码部分填充为1e8，并对数组进行形状变换，转换为二维数组，并计算每行的最小值
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 将x_min、y_min、x_max、y_max四个数组按列合并成一个数组，并返回
    return np.stack([x_min, y_min, x_max, y_max], 1)
# 从transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation 复制代码，将DETR->ConditionalDetr
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
    # 获取注释文件路径
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    # 创建一个新的目标字典
    new_target = {}
    # 设置图像ID
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 设置图像尺寸
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    # 检查目标是否包含分段信息
    if "segments_info" in target:
        # 读取注释文件中的掩码
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将RGB格式的掩码转换为ID格式
        masks = rgb_to_id(masks)

        # 获取分段信息中的ID
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 根据ID创建掩码
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        # 如果需要返回掩码，则存储在新目标字典中
        if return_masks:
            new_target["masks"] = masks
        # 根据掩码生成边界框
        new_target["boxes"] = masks_to_boxes(masks)
        # 获取类别标签
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 获取是否是群体类标签
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        # 获取区域面积
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


# 从transformers.models.detr.image_processing_detr.get_segmentation_image 复制代码
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入图像的高度和宽度
    h, w = input_size
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size

    # 对掩码进行softmax操��
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果掩码维度为0，则设置所有元素为0
    if m_id.shape[-1] == 0:
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取最大值的掩码ID
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重
    if deduplicate:
        # 将属于同一类的掩码合并
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将ID格式掩码转换为RGB格式
    seg_img = id_to_rgb(m_id)
    # 调整图像大小为最终大小
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# 从transformers.models.detr.image_processing_detr.get_mask_area 复制代码
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    final_h, final_w = target_size
    # 将掩码图像转换为无符号8位整数
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 将 RGB 分割图像转换为类别 ID 图像
    m_id = rgb_to_id(np_seg_img)
    # 计算每个类别的像素数量，存储在列表中
    area = [(m_id == i).sum() for i in range(n_classes)]
    # 返回每个类别的像素数量列表
    return area
# 从类别概率 logits 中计算标签得分
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 将 logits 转换为概率值
    probs = scipy.special.softmax(logits, axis=-1)
    # 获取概率最大的类别作为标签
    labels = probs.argmax(-1, keepdims=True)
    # 从概率中取出对应标签的得分
    scores = np.take_along_axis(probs, labels, axis=-1)
    # 去掉无用的维度
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# 从 ConditionalDetrForSegmentation 输出结果中生成全景分割预测
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
    将 ConditionalDetrForSegmentation 的输出转换成单个样本的全景分割预测

    Args:
        out_logits (`torch.Tensor`): 该样本的 logits
        masks (`torch.Tensor`): 预测的分割掩码
        boxes (`torch.Tensor`): 预测的边界框，归一化格式为 `(center_x, center_y, width, height)`，值区间为 `[0, 1]`
        processed_size (`Tuple[int, int]`): 图像处理后的尺寸 `(height, width)`
        target_size (`Tuple[int, int]`): 图像的目标尺寸 `(height, width)`
        is_thing_map (`Dict`): 将类别索引映射到布尔值，指示类别是否为物体
        threshold (`float`, *可选*, 默认为 0.85): 用于二值化分割掩码的阈值
    """
    # 过滤空查询和低于阈值的检测
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

    # 可能存在多个相同类别的预测掩码，以下跟踪每个类别掩码的列表（稍后会合并）
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    # 对于当前的每一个类别，在是否为实物类别的映射中使用枚举函数进行遍历，并将不是实物类别的索引加入到对应的列表中
    for k, label in enumerate(cur_classes):
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)
    
    # 根据当前的掩膜，获取分割图像
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    # 获取每个掩膜的面积
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))
    
    # 过滤掉面积较小的掩膜
    if cur_classes.size() > 0:
        # 只要还有小于等于4的掩膜，就过滤掉它们
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        while filtered_small.any():
            # 去掉小面积的掩膜
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            # 根据当前的掩膜，获取分割图像
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            # 获取每个掩膜的面积
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        # 如果没有掩膜，将当前类别设置为1
        cur_classes = np.ones((1, 1), dtype=np.int64)
    
    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    # 删除cur_classes
    del cur_classes
    
    # 使用io.BytesIO作为输出流
    with io.BytesIO() as out:
        # 将seg_img转换为PIL.Image，保存为PNG格式并写入到输出流中
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        # 将预测结果保存为字典形式，包含png图片的二进制字符串和分割信息
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
    
    # 返回预测结果
    return predictions
# 从transformers.models.detr.image_processing_detr.resize_annotation复制
# 将注释调整大小为目标大小
def resize_annotation(
    annotation: Dict[str, Any],
    # 注释字典
    orig_size: Tuple[int, int],
    # 输入图像的原始大小
    target_size: Tuple[int, int],
    # 图像的目标大小，预处理后返回的大小
    threshold: float = 0.5,
    # 阈值用于对分割遮罩进行二值化
    resample: PILImageResampling = PILImageResampling.NEAREST,
    # 调整遮罩时使用的重采样滤波器
):
# 省略了具体实现内容


# 从transformers.models.detr.image_processing_detr.binary_mask_to_rle复制
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
    # 如果是torch张量，则转换为numpy数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将掩码展平
    pixels = mask.flatten()
    # 将0添加到掩码前面和后面
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到像素值变化的索引
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 计算运行的长度
    runs[1::2] -= runs[::2]
    # 返回运行长度编码的列表
    return list(runs)


# 从transformers.models.detr.image_processing_detr.convert_segmentation_to_rle复制
# 将分割转换为运行长度编码
def convert_segmentation_to_rle(segmentation):
    # 省略了具体实现内容
    # 将给定的形状为 `(height, width)` 的分割地图转换成行长度编码（RLE）格式
    # 参数：
    #   segmentation (`torch.Tensor` or `numpy.array`):
    #       形状为 `(height, width)` 的分割地图，其中每个值表示一个片段或类别 ID
    # 返回值：
    #   `List[List]`: 一个列表，其中每个列表是一个分段/类别 ID 的行长度编码
    
    # 获取分割地图中唯一的段落 ID
    segment_ids = torch.unique(segmentation)
    
    # 用于存储行长度编码的空列表
    run_length_encodings = []
    
    # 对于每个段落 ID，获取对应的掩码，并将其转换为行长度编码
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)
    
    # 返回行长度编码列表
    return run_length_encodings
# 从 transformers.models.detr.image_processing_detr.remove_low_and_no_objects 复制的函数，用于移除低分数和无对象的掩码
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    使用 `object_mask_threshold` 进行二值化给定的掩码，返回相关的 `masks`、`scores` 和 `labels` 值。

    Args:
        masks (`torch.Tensor`):
            形状为 `(num_queries, height, width)` 的张量。
        scores (`torch.Tensor`):
            形状为 `(num_queries)` 的张量。
        labels (`torch.Tensor`):
            形状为 `(num_queries)` 的张量。
        object_mask_threshold (`float`):
            介于 0 和 1 之间的数字，用于二值化掩码。
    Raises:
        `ValueError`: 当所有输入张量的第一个维度不匹配时抛出异常。
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: 所有地区小于 `object_mask_threshold` 的 `masks`、`scores` 和 `labels`。
    """

    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# 从 transformers.models.detr.image_processing_detr.check_segment_validity 复制的函数，用于检查分段的有效性
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第 k 类关联的掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询 k 中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除不连通的小片段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 从 transformers.models.detr.image_processing_detr.compute_segments 复制的函数，用于计算分段
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

    # 按其预测分数对每个掩码进行加权
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    # 创建一个空字典，用于存储预测标签对应的对象分割 ID
    stuff_memory_list: Dict[str, int] = {}

    # 遍历预测标签的数量
    for k in range(pred_labels.shape[0]):
        # 获取预测的类别标签
        pred_class = pred_labels[k].item()
        # 检查当前类别是否需要融合
        should_fuse = pred_class in label_ids_to_fuse

        # 检查是否存在分割掩码，并且足够大以用于对象分割
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        # 如果存在分割掩码
        if mask_exists:
            # 如果当前类别已经在 stuff_memory_list 中
            if pred_class in stuff_memory_list:
                # 获取当前类别的对象分割 ID
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 否则，递增当前类别的对象分割 ID
                current_segment_id += 1

            # 将当前对象分割添加到最终的分割图中
            segmentation[mask_k] = current_segment_id
            # 获取当前对象分割的得分
            segment_score = round(pred_scores[k].item(), 6)
            # 将当前对象分割的信息添加到 segments 列表中
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            # 如果应该对当前类别进行融合，则更新 stuff_memory_list 中的对象分割 ID
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    # 返回最终的分割图和对象分割信息列表
    return segmentation, segments
class ConditionalDetrImageProcessor(BaseImageProcessor):
    r"""
    构造一个 Conditional Detr 图像处理器。

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            注解的数据格式。可以是 "coco_detection" 或 "coco_panoptic" 之一。
        do_resize (`bool`, *optional*, defaults to `True`):
            控制是否将图像的 (height, width) 维度调整为指定的 `size`。可以被 `preprocess` 方法中的 `do_resize` 参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            调整大小后的图像 (height, width) 维度。可以被 `preprocess` 方法中的 `size` 参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            如果调整图像大小，则使用的重采样滤波器。
        do_rescale (`bool`, *optional*, defaults to `True`):
            控制是否按指定比例 `rescale_factor` 进行重新缩放图像。可以被 `preprocess` 方法中的 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            重新缩放图像时使用的缩放因子。可以被 `preprocess` 方法中的 `rescale_factor` 参数覆盖。
        do_normalize:
            控制是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            归一化图像时使用的平均值。可以是单个值或一个值列表，每个通道一个。可以被 `preprocess` 方法中的 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            归一化图像时使用的标准差值。可以是单个值或一个值列表，每个通道一个。可以被 `preprocess` 方法中的 `image_std` 参数覆盖。
        do_pad (`bool`, *optional*, defaults to `True`):
            控制是否将图像填充到批处理中最大的图像，并创建像素掩码。可以被 `preprocess` 方法中的 `do_pad` 参数覆盖。
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.__init__ 复制而来
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,  # 初始化方法，设置参数format，默认值为COCO_DETECTION
        do_resize: bool = True,  # 是否调整大小，默认为True
        size: Dict[str, int] = None,  # 图像尺寸的字典，包括最短边和最长边，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_rescale: bool = True,  # 是否重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否归一化，默认为True
        image_mean: Union[float, List[float]] = None,  # 图像均值，默认为None
        image_std: Union[float, List[float]] = None,  # 图像标准差，默认为None
        do_pad: bool = True,  # 是否填充，默认为True
        **kwargs,  # 其他参数
    ) -> None:  # 返回空值
        if "pad_and_return_pixel_mask" in kwargs:  # 如果kwargs中包含"pad_and_return_pixel_mask"
            do_pad = kwargs.pop("pad_and_return_pixel_mask")  # 将"pad_and_return_pixel_mask"弹出，并赋值给do_pad

        if "max_size" in kwargs:  # 如果kwargs中包含"max_size"
            logger.warning_once(  # 记录警告日志
                "The `max_size` parameter is deprecated and will be removed in v4.26. "  # 警告信息
                "Please specify in `size['longest_edge'] instead`.",  # 提示信息
            )
            max_size = kwargs.pop("max_size")  # 将"max_size"弹出，并赋值给max_size
        else:  # 否则
            max_size = None if size is None else 1333  # 如果size为None，则max_size为None，否则为1333

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}  # 如果size不为None，则保持不变，否则设为默认值
        size = get_size_dict(size, max_size=max_size, default_to_square=False)  # 获取最终的尺寸字典

        super().__init__(**kwargs)  # 调用父类初始化方法
        self.format = format  # 格式
        self.do_resize = do_resize  # 是否调整大小
        self.size = size  # 尺寸
        self.resample = resample  # 重采样方法
        self.do_rescale = do_rescale  # 是否重新缩放
        self.rescale_factor = rescale_factor  # 重新缩放因子
        self.do_normalize = do_normalize  # 是否归一化
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN  # 图像均值
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD  # 图像标准差
        self.do_pad = do_pad  # 是否填充

    @classmethod  # 类方法修饰符
    # 从字典创建对象，修改自transformers.models.detr.image_processing_detr.DetrImageProcessor.from_dict，将DETR改为ConditionalDetr
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `ConditionalDetrImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()  # 复制输入的字典
        if "max_size" in kwargs:  # 如果kwargs中包含"max_size"
            image_processor_dict["max_size"] = kwargs.pop("max_size")  # 将kwargs中的"max_size"弹出并赋值给image_processor_dict
        if "pad_and_return_pixel_mask" in kwargs:  # 如果kwargs中包含"pad_and_return_pixel_mask"
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")  # 将kwargs中的"pad_and_return_pixel_mask"弹出并赋值给image_processor_dict
        return super().from_dict(image_processor_dict, **kwargs)  # 调用父类的from_dict方法

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_annotation复制，将DETR改为ConditionalDetr
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个方法，准备用于 ConditionalDetr 模型的标注数据
    def prepare_annotation(self, image, target, return_segmentation_masks=None, masks_path=None, format=None) -> Dict:
        """
        Prepare an annotation for feeding into ConditionalDetr model.
        """
        # 如果未指定格式，则使用默认格式
        format = format if format is not None else self.format

        # 如果格式是 COCO_DETECTION
        if format == AnnotationFormat.COCO_DETECTION:
            # 如果未指定是否返回分割掩模，则默认不返回
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO 检测标注
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        # 如果格式是 COCO_PANOPTIC
        elif format == AnnotationFormat.COCO_PANOPTIC:
            # 如果未指定是否返回分割掩模，则默认返回
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 准备 COCO 全景标注
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        # 如果格式不支持，抛出 ValueError 异常
        else:
            raise ValueError(f"Format {format} is not supported.")
        # 返回标注数据
        return target

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare 复制而来
    # 此方法已弃用，将在 v4.33 版本中移除，建议使用 prepare_annotation 方法代替
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 使用 prepare_annotation 方法准备标注数据，不再返回图像
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回图像和标注数据
        return image, target

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.convert_coco_poly_to_mask 复制而来
    # 此方法已弃用，将在 v4.33 版本中移除
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用 convert_coco_poly_to_mask 方法
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_detection 复制而来
    # 此方法已弃用，将在 v4.33 版本中移除
    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用 prepare_coco_detection_annotation 方法
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.prepare_coco_panoptic 复制而来
    # 此方法已弃用，将在 v4.33 版本中移除
    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用 prepare_coco_panoptic_annotation 方法
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.resize 复制而来
    # 定义一个resize函数，用于缩放图片到指定大小
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

        # 如果参数中包含 "max_size"，则发出警告并将其弹出
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # 获取指定大小的字典，并不默认为正方形
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 根据指定的大小和格式进行图片缩放
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果大小字典不包含指定的键，则抛出数值错误
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # 对图片进行缩放
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        # 返回被缩放后的图片
        return image

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation复制而来
    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    # 定义一个函数，将标注调整为与调整后的图像匹配。如果大小是一个整数，标注的较小边将匹配到这个数字。
    def resize_annotation(
        annotation: Dict,
        orig_size: Tuple[int, int],
        size: Union[int, Tuple[int, int]],
        resample: Optional[Union[int, str]] = None
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale中复制而来
    # 定义一个函数，通过给定的因子重新缩放图像。image = image * rescale_factor.
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

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation中复制而来
    # 定义一个函数，将标注中的框从`[top_left_x, top_left_y, bottom_right_x, bottom_right_y]`格式规范化为`[center_x, center_y, width, height]`格式。
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format.
        """
        return normalize_annotation(annotation, image_size=image_size)

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image中复制而来
    # 定义一个函数，填充图像以匹配输出大小
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个函数，用于对图像进行零填充至指定大小
    def pad(self, images: List[np.ndarray], constant_values: Union[float, Iterable[float]] = 0, return_pixel_mask: bool = True, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = None, input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
        # 构建填充元组，指定要在顶部、底部、左侧和右侧分别填充的像素数
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

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
    # 定义填充函数，用于对图像进行填充操作
    # 参数包括：图像列表、填充常数值、是否返回像素掩码、返回的张量类型、数据格式和输入数据格式
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
        # 获取 images 中最大高度和宽度，并用于填充
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
        # 组装填充后的像素值
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 为每个图像创建像素掩码
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 组装像素掩码
            data["pixel_mask"] = masks

        # 返回填充后的图像和相关数据
        return BatchFeature(data=data, tensor_type=return_tensors)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.preprocess
    # 定义一个预处理方法，用于处理图像和注释数据
    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        format: Optional[Union[str, AnnotationFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        # 此方法用于预处理图像和对应的注释数据，可以接受多种参数，包括输入图像、注释、调整尺寸、归一化等
        # images: 输入的图像数据，可以是单个图像或图像列表
        # annotations: 对应的标注数据，可以是单个标注或标注列表，可选参数
        # return_segmentation_masks: 是否返回分割掩码，布尔值，可选参数
        # masks_path: 分割掩码保存路径，字符串或路径对象，可选参数
        # do_resize: 是否进行调整图像大小，布尔值，可选参数
        # size: 调整后的图像大小，字典形式，可选参数
        # resample: 调整图像大小的插值方法，PILImageResampling 枚举类型，可选参数
        # do_rescale: 是否进行图像尺度缩放，布尔值，可选参数
        # rescale_factor: 图像尺度缩放因子，整数或浮点数，可选参数
        # do_normalize: 是否进行图像归一化，布尔值，可选参数
        # image_mean: 图像归一化的均值，单个值或列表，可选参数
        # image_std: 图像归一化的标准差，单个值或列表，可选参数
        # do_pad: 是否进行图像填充，布尔值，可选参数
        # format: 图像和注释数据的格式，字符串或注释格式枚举，可选参数
        # return_tensors: 返回数据的格式，张量类型或字符串，可选参数
        # data_format: 数据的通道顺序，字符串或通道维度枚举，默认为第一通道，可选参数
        # input_data_format: 输入数据的通道顺序，字符串或通道维度枚举，可选参数
        # **kwargs: 其他可选参数，用于灵活扩展
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
        logging.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # Extract the logits and predicted boxes from the model outputs
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # Check if the number of logits is equal to the number of target sizes
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # Check if each element of target_sizes contains the size (h, w) of each image of the batch
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # Calculate the probability and pick the top 300 values and their indexes
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        # Convert the predicted boxes from center format to corners format
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Convert the relative [0, 1] coordinates to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Create a list of dictionaries containing scores, labels, and boxes for each image in the batch
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    # Copied from transformers.models.deformable_detr.image_processing_deformable_detr.DeformableDetrImageProcessor.post_process_object_detection with DeformableDetr->ConditionalDetr
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None, top_k: int = 100
    ):
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
        # 获取模型输出中的分类得分和边界框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 如果指定了目标大小，则进行检查
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # 对分类得分进行 sigmoid 操作
        prob = out_logits.sigmoid()
        # 将概率张量形状转换为 (batch_size, num_boxes)
        prob = prob.view(out_logits.shape[0], -1)
        # 保留前 top_k 个预测框
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        # 保存 top_k 的得分
        scores = topk_values
        # 计算 top_k 的边界框索引，并转换成绝对坐标
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        # 提取类别标签
        labels = topk_indexes % out_logits.shape[2]
        # 将中心点表示的边界框转换成左上角和右下角坐标表示
        boxes = center_to_corners_format(out_bbox)
        # 从所有边界框中选出 top_k 的边界框
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 将相对坐标转换为绝对坐标
        if isinstance(target_sizes, List):
            # 如果目标大小是列表，则将列表中的每个目标大小拆解为高度和宽度
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            # 否则，从目标大小张量中获取高度和宽度
            img_h, img_w = target_sizes.unbind(1)
        # 计算比例因子并应用到边界框上
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 将结果组织成字典列表形式
        results = []
        for s, l, b in zip(scores, labels, boxes):
            # 根据阈值过滤得分低于阈值的预测结果
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_semantic_segmentation 复制并修改为 ConditionalDetr
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

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_instance_segmentation with Detr->ConditionalDetr
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,

The above code includes two methods: `post_process_semantic_segmentation` and `post_process_instance_segmentation`. 

The first method `post_process_semantic_segmentation` converts the output of `ConditionalDetrForSegmentation` into semantic segmentation maps. It takes raw model outputs and a list of target sizes as input, and returns a list of tensors representing semantic segmentation maps.

The second method `post_process_instance_segmentation` is copied from another source with `Detr` replaced by `ConditionalDetr`. It takes various parameters as input and seems to be related to processing instance segmentation, but the rest of the functionality is not shown in the provided code.
    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_panoptic_segmentation复制过来，在Detr->ConditionalDetr
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 阈值，用于筛选输出结果
        mask_threshold: float = 0.5,  # 遮罩阈值，用于筛选遮罩
        overlap_mask_area_threshold: float = 0.8,  # 重叠遮罩区域的阈值
        label_ids_to_fuse: Optional[Set[int]] = None,  # 需要融合的标签ID集合，可选
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标大小的列表，可选
```