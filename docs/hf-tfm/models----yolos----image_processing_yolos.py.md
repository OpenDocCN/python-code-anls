# `.\models\yolos\image_processing_yolos.py`

```
    # 计算按照指定尺寸和最大尺寸缩放后的图像大小，保持宽高比不变
    """
    Compute the size of the image while maintaining the aspect ratio based on the given size and optional maximum size.
    """
    aspect_ratio = float(image_size[0]) / image_size[1]
    # 如果没有提供最大尺寸或者图像尺寸在最大尺寸内，则直接返回图像尺寸
    if max_size is None or (size[0] <= max_size[0] and size[1] <= max_size[1]):
        return size
    # 根据宽高比计算缩放后的高度
    new_height = int(round(size[0] / aspect_ratio))
    # 如果新高度小于等于最大高度，返回结果
    if new_height <= max_size[0]:
        return size[0], new_height
    # 否则，根据宽高比计算缩放后的宽度，并返回结果
    new_width = int(round(size[1] * aspect_ratio))
    return new_width, size[1]
    # 获取输入图像的高度和宽度
    height, width = image_size

    # 如果设置了最大输出尺寸
    if max_size is not None:
        # 计算输入图像的最小和最大边长
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        
        # 如果按照指定输出尺寸计算后超过了最大允许尺寸，则重新调整输出尺寸
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    # 如果宽度小于高度且宽度不等于指定尺寸，则调整高度以保持比例
    if width < height and width != size:
        height = int(size * height / width)
        width = size
    # 如果高度小于宽度且高度不等于指定尺寸，则调整宽度以保持比例
    elif height < width and height != size:
        width = int(size * width / height)
        height = size

    # 计算宽度的模数，以确保宽度为16的倍数
    width_mod = np.mod(width, 16)
    # 计算高度的模数，以确保高度为16的倍数
    height_mod = np.mod(height, 16)

    # 调整宽度和高度，使其成为16的倍数
    width = width - width_mod
    height = height - height_mod

    # 返回调整后的高度和宽度作为元组
    return (height, width)
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
    # 如果输出尺寸是一个元组或列表，则直接返回该尺寸
    if isinstance(size, (list, tuple)):
        return size
    # 否则，根据输入图像尺寸的长宽比计算输出尺寸
    return get_size_with_aspect_ratio(image_size, size, max_size)


# Copied from transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    # 如果输入是一个 numpy 数组，则返回 numpy 的 array 函数
    if isinstance(arr, np.ndarray):
        return np.array
    # 如果 TensorFlow 可用且输入是 TensorFlow 张量，则返回 TensorFlow 的 convert_to_tensor 函数
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    # 如果 PyTorch 可用且输入是 PyTorch 张量，则返回 PyTorch 的 tensor 函数
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    # 如果 Flax 可用且输入是 JAX 张量，则返回 JAX 的 array 函数
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    # 如果无法识别输入类型，则引发 ValueError 异常
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# Copied from transformers.models.detr.image_processing_detr.safe_squeeze
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    # 如果未指定轴，则默认压缩所有维度为 1 的轴
    if axis is None:
        return arr.squeeze()
    # 否则，尝试压缩指定轴，若指定轴的维度不为 1 则返回原数组
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# Copied from transformers.models.detr.image_processing_detr.normalize_annotation
def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    norm_annotation = {}
    # 遍历注释字典中的键值对
    for key, value in annotation.items():
        # 如果当前键是 "boxes"
        if key == "boxes":
            # 将值赋给变量 boxes
            boxes = value
            # 将边角格式的框转换为中心-大小格式的框
            boxes = corners_to_center_format(boxes)
            # 将框的坐标值除以图像宽度和高度，进行归一化处理
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            # 将归一化后的框重新赋给规范化后的注释中的 "boxes" 键
            norm_annotation[key] = boxes
        else:
            # 对于非 "boxes" 键，直接复制其值到规范化后的注释中
            norm_annotation[key] = value
    # 返回处理后的规范化注释字典
    return norm_annotation
# 从transformers.models.detr.image_processing_detr.max_across_indices复制而来
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    返回可迭代值中所有索引上的最大值列表。
    """
    # 对于可迭代值的每个索引，找到最大值并构成列表返回
    return [max(values_i) for values_i in zip(*values)]


# 从transformers.models.detr.image_processing_detr.make_pixel_mask复制而来
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    为图像生成像素掩码，其中1表示有效像素，0表示填充。

    Args:
        image (`np.ndarray`):
            需要生成像素掩码的图像。
        output_size (`Tuple[int, int]`):
            掩码的输出尺寸。
    """
    # 获取图像的高度和宽度，根据输入的数据格式
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个指定大小的全零数组
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效像素的部分设置为1
    mask[:input_height, :input_width] = 1
    return mask


# 从transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask复制而来
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    将COCO格式的多边形注释转换为掩码。

    Args:
        segmentations (`List[List[float]]`):
            多边形的列表，每个多边形由一组x-y坐标表示。
        height (`int`):
            掩码的高度。
        width (`int`):
            掩码的宽度。
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools未安装在您的环境中。")

    masks = []
    # 遍历每个多边形，将其转换为掩码
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    # 如果存在掩码，则堆叠它们成为一个数组
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks


# 从transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation复制而来
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    将COCO格式中的目标转换为DETR所期望的格式。
    """
    # 获取图像的高度和宽度，根据输入的数据格式
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # 获取给定图像的所有COCO注释
    annotations = target["annotations"]
    # 过滤掉所有iscrowd为0或未定义的对象
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # 获取所有对象的类别ID
    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # 用于转换为coco api的准备工作
    # 提取所有注释中的目标区域面积，转换为 numpy 数组，使用 float32 类型
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    # 提取所有注释中的是否为群体标志（如果存在），转换为 numpy 数组，如果不存在则设为0，使用 int64 类型
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    # 提取所有注释中的包围框（bbox），存入列表
    boxes = [obj["bbox"] for obj in annotations]
    # 防止出现没有包围框的情况，通过调整大小使所有框都有四个坐标值
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # 将宽度和高度加到每个框的坐标上，使其成为 (left, top, right, bottom) 格式
    boxes[:, 2:] += boxes[:, :2]
    # 将框的左上角和右下角坐标限制在图像尺寸内，即不超过图像宽度和高度
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    # 通过检查每个框的高度和宽度来保留有效的框
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    # 创建新的目标字典
    new_target = {}
    # 将图像 ID 添加到新的目标字典中
    new_target["image_id"] = image_id
    # 将保留框对应的类标签添加到新的目标字典中
    new_target["class_labels"] = classes[keep]
    # 将保留框添加到新的目标字典中
    new_target["boxes"] = boxes[keep]
    # 将保留框对应的区域面积添加到新的目标字典中
    new_target["area"] = area[keep]
    # 将保留框对应的是否为群体标志添加到新的目标字典中
    new_target["iscrowd"] = iscrowd[keep]
    # 将原始图像尺寸作为整数数组添加到新的目标字典中
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    # 如果存在注释并且第一个注释包含关键点信息
    if annotations and "keypoints" in annotations[0]:
        # 提取所有注释中的关键点列表
        keypoints = [obj["keypoints"] for obj in annotations]
        # 将过滤后的关键点列表转换为 numpy 数组
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # 使用 keep 掩码过滤出相关的注释
        keypoints = keypoints[keep]
        # 计算关键点的数量
        num_keypoints = keypoints.shape[0]
        # 如果有关键点，将其重新整形为 (-1, 3) 的形状，否则保持原状
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        # 将关键点添加到新的目标字典中
        new_target["keypoints"] = keypoints

    # 如果需要返回分割掩模
    if return_segmentation_masks:
        # 提取所有注释中的分割信息列表
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        # 将 COCO 多边形分割转换为掩模（masks）
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        # 将保留的掩模添加到新的目标字典中
        new_target["masks"] = masks[keep]

    # 返回最终构建的新目标字典
    return new_target
# Copied from transformers.models.detr.image_processing_detr.masks_to_boxes
# 计算提供的全景分割掩模周围的边界框
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    # 如果掩模为空，则返回一个形状为 (0, 4) 的全零数组
    if masks.size == 0:
        return np.zeros((0, 4))

    # 获取掩模的高度和宽度
    h, w = masks.shape[-2:]
    # 创建高度和宽度的一维数组
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # 创建高度和宽度的二维数组网格，使用 "ij" 索引顺序
    y, x = np.meshgrid(y, x, indexing="ij")

    # 计算掩模和坐标 x 的乘积，并获取每个掩模的最大 x 坐标
    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    # 创建掩模的 x 坐标的掩码数组，处理不包含掩模的部分
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    # 将掩码数组填充为 1e8，获取每个掩模的最小 x 坐标
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    # 计算掩模和坐标 y 的乘积，并获取每个掩模的最大 y 坐标
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    # 创建掩模的 y 坐标的掩码数组，处理不包含掩模的部分
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    # 将掩码数组填充为 1e8，获取每个掩模的最小 y 坐标
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    # 返回形状为 `[number_masks, 4]` 的边界框数组，包含每个掩模的最小 x、y 和最大 x、y 坐标
    return np.stack([x_min, y_min, x_max, y_max], 1)


# Copied from transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation with DETR->YOLOS
# 为 YOLOS 准备 coco 全景注释
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for YOLOS.
    """
    # 获取图像的高度和宽度，使用输入数据格式作为通道维度
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    # 构建注释路径，结合掩模路径和目标文件名
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    # 创建新的目标字典，包含图像 ID、大小和原始大小的信息
    new_target = {}
    # 使用目标中的图像 ID 或 ID，作为 64 位整数数组的一部分
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    # 图像的尺寸，作为 64 位整数数组的一部分
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 原始图像的尺寸，作为 64 位整数数组的一部分
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)
    # 检查目标字典中是否存在键名为 "segments_info"
    if "segments_info" in target:
        # 从指定路径打开图像文件，将其转换为 NumPy 数组形式的多通道图像数据
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        # 将 RGB 图像数据转换为整数形式的类别 ID 图像数据
        masks = rgb_to_id(masks)

        # 从目标字典中获取所有段落信息的 ID 组成的数组
        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        # 根据 ID 数组，创建布尔类型的掩码数组，用于表示每个像素是否属于对应的类别
        masks = masks == ids[:, None, None]
        # 将布尔类型的掩码数组转换为整数类型（0 或 1）
        masks = masks.astype(np.uint8)

        # 如果需要返回掩码数组，则将其添加到新的目标字典中
        if return_masks:
            new_target["masks"] = masks
        
        # 将掩码数组转换为包围框信息并添加到新的目标字典中
        new_target["boxes"] = masks_to_boxes(masks)
        
        # 从段落信息中获取类别 ID 并转换为 NumPy 数组形式，添加到新的目标字典中
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        
        # 从段落信息中获取 iscrowd 属性并转换为 NumPy 数组形式，添加到新的目标字典中
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        
        # 从段落信息中获取 area 属性并转换为 NumPy 数组形式，添加到新的目标字典中
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    # 返回经处理后的新目标字典
    return new_target
# Copied from transformers.models.detr.image_processing_detr.get_segmentation_image
def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    # 获取输入图像的高度和宽度
    h, w = input_size
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size

    # 对 mask 进行 softmax 操作，使得每个像素的概率和为 1
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    # 如果没有检测到任何 mask，则将 m_id 初始化为全零数组
    if m_id.shape[-1] == 0:
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        # 取最大概率对应的类别作为每个像素的预测标签，并重新形状为 (h, w)
        m_id = m_id.argmax(-1).reshape(h, w)

    # 如果需要去重复处理
    if deduplicate:
        # 合并相同类别的 mask
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    # 将预测标签转换为 RGB 彩色图像
    seg_img = id_to_rgb(m_id)
    # 将彩色分割图像 resize 到最终的目标大小
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# Copied from transformers.models.detr.image_processing_detr.get_mask_area
def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    # 获取目标图像的最终高度和宽度
    final_h, final_w = target_size
    # 将 seg_img 转换为 uint8 类型的 numpy 数组，并重塑形状为 (final_h, final_w, 3)
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    # 将 RGB 彩色图像转换为类别标签图像
    m_id = rgb_to_id(np_seg_img)
    # 计算每个类别的面积
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area


# Copied from transformers.models.detr.image_processing_detr.score_labels_from_class_probabilities
def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 对类别概率进行 softmax 操作
    probs = scipy.special.softmax(logits, axis=-1)
    # 取概率最大的类别作为预测标签
    labels = probs.argmax(-1, keepdims=True)
    # 提取最大概率值作为预测得分，并去除冗余的维度
    scores = np.take_along_axis(probs, labels, axis=-1)
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


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
    # 计算原始大小与目标大小的尺寸比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    # 创建新的注释字典，并设置目标尺寸大小
    new_annotation = {}
    new_annotation["size"] = target_size
    # 遍历注释字典中的每个键值对
    for key, value in annotation.items():
        # 如果键是"boxes"
        if key == "boxes":
            # 将值赋给变量boxes，并按比例缩放每个框的坐标
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            # 将缩放后的框坐标存入新注释字典中
            new_annotation["boxes"] = scaled_boxes
        # 如果键是"area"
        elif key == "area":
            # 将值赋给变量area，并按比例缩放面积
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            # 将缩放后的面积存入新注释字典中
            new_annotation["area"] = scaled_area
        # 如果键是"masks"
        elif key == "masks":
            # 将值赋给变量masks，并按目标尺寸重新调整每个掩码，然后进行二值化处理
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            # 将处理后的掩码存入新注释字典中
            new_annotation["masks"] = masks
        # 如果键是"size"
        elif key == "size":
            # 将目标尺寸存入新注释字典中
            new_annotation["size"] = target_size
        # 对于其它未指定处理的键，直接复制值到新注释字典中
        else:
            new_annotation[key] = value

    # 返回更新后的新注释字典
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
    # 如果输入的 mask 是 PyTorch tensor，则转换为 numpy 数组
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将二维数组扁平化为一维数组
    pixels = mask.flatten()
    # 在数组两端各加一个 0，确保算法正确处理首尾边界
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到连续变化的位置，形成 RLE 编码
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 计算每段长度并调整格式
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
    # 获取所有不同的 segment_id
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    # 遍历每个 segment_id，生成对应的 RLE 编码
    for idx in segment_ids:
        # 根据当前 segment_id 构建对应的二进制 mask
        mask = torch.where(segmentation == idx, 1, 0)
        # 将二进制 mask 转换为 RLE 编码
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

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
    # 检查输入的张量形状是否一致
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    # 生成一个布尔掩码，标识需要保留的对象
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    # 根据掩码筛选出需要保留的 masks, scores 和 labels，并返回
    return masks[to_keep], scores[to_keep], labels[to_keep]


# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
# 检查分割有效性的函数，判断给定类别 k 的分割是否有效
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与类别 k 相关联的掩模
    mask_k = mask_labels == k
    # 计算类别 k 的掩模面积
    mask_k_area = mask_k.sum()

    # 计算查询类别 k 中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 判断是否存在类别 k 的掩模
    mask_exists = mask_k_area > 0 and original_area > 0

    # 如果存在掩模，进一步检查是否是有效的分割
    if mask_exists:
        # 计算掩模面积与查询面积的比率
        area_ratio = mask_k_area / original_area
        # 如果比率不大于重叠掩模面积阈值，将掩模标记为无效
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
    # 确定图像的高度和宽度
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 创建用于存储分割结果的空白分割图和段落列表
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    # 如果指定了目标大小，则对掩模进行插值
    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # 根据预测分数对每个掩模进行加权处理
    mask_probs *= pred_scores.view(-1, 1, 1)
    # 获取掩模的标签，即每个像素最可能的类别
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别实例的数量
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查是否存在有效的分割掩模
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            # 如果类别已经存在于 stuff_memory_list 中，则使用已有的分割 ID
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                # 否则递增分割 ID
                current_segment_id += 1

            # 将当前对象的分割添加到最终的分割图中
            segmentation[mask_k] = current_segment_id
            # 获取分割的分数，并将分割信息添加到 segments 列表中
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            # 如果应该进行融合，则将类别 ID 与当前分割 ID 关联起来
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


class YolosImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Detr image processor.
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
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True` will pad the images in the batch to the largest height and width in the batch.
            Padding will be applied to the bottom and right of the image with zeros.
    """

    # 定义模型输入的名称列表
    model_input_names = ["pixel_values", "pixel_mask"]
    # 初始化方法，用于创建一个新的对象实例
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
        # 如果在 kwargs 中有 "pad_and_return_pixel_mask" 参数，则使用该参数覆盖 do_pad 变量
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        # 如果在 kwargs 中有 "max_size" 参数，则发出警告并将其从 kwargs 中弹出
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            # 否则将 max_size 设置为 None，或者从 size 中获取默认值 1333
            max_size = None if size is None else 1333

        # 如果 size 为 None，则设置一个默认的 size 字典，包含 shortest_edge 和 longest_edge
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # 调用 get_size_dict 方法获取最终的 size 字典，考虑 max_size 和 default_to_square 参数
        size = get_size_dict(size, max_size=max_size, default_to_square=False)

        # 兼容性处理：如果 do_convert_annotations 为 None，则将其设为 do_normalize 的值
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置实例的各种属性
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
        # 定义一个私有属性，包含所有有效的处理器键名
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
            "image_mean",
            "image_std",
            "do_convert_annotations",
            "do_pad",
            "format",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.from_dict 方法复制而来，修改为支持 Yolos
    # 从字典中恢复图像处理器对象的参数，并根据传入的 kwargs 更新参数。
    # 如果 kwargs 中包含 "max_size"，则更新 image_processor_dict 中的 "max_size"。
    # 如果 kwargs 中包含 "pad_and_return_pixel_mask"，则更新 image_processor_dict 中的 "pad_and_return_pixel_mask"。
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        image_processor_dict = image_processor_dict.copy()
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        return super().from_dict(image_processor_dict, **kwargs)

    # 从 DETR 模型的图像处理器中准备注释信息，以便输入到 DETR 模型中。
    # 根据指定的注释格式进行处理。
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
        Prepare an annotation for feeding into DETR model.
        """
        format = format if format is not None else self.format

        # 根据注释格式选择相应的处理方法：COCO_DETECTION 或 COCO_PANOPTIC。
        if format == AnnotationFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_detection_annotation 函数处理 COCO_DETECTION 格式的注释。
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        elif format == AnnotationFormat.COCO_PANOPTIC:
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            # 调用 prepare_coco_panoptic_annotation 函数处理 COCO_PANOPTIC 格式的注释。
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            # 如果注释格式不是 COCO_DETECTION 或 COCO_PANOPTIC，则抛出 ValueError。
            raise ValueError(f"Format {format} is not supported.")
        return target

    # 警告：`prepare` 方法已弃用，将在 v4.33 版本中删除。请使用 `prepare_annotation` 方法替代。
    # 注意：`prepare_annotation` 方法不再返回图像。
    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once(
            "The `prepare` method is deprecated and will be removed in a v4.33. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        # 调用 prepare_annotation 方法处理注释。
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        # 返回处理后的图像和目标。
        return image, target

    # 从 DETR 模型的图像处理器中将 COCO 格式的多边形转换为掩码的方法。
    # 发出警告日志，指出方法 `convert_coco_poly_to_mask` 将在 v4.33 版本中删除
    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once("The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ")
        # 调用同名函数 `convert_coco_poly_to_mask` 处理传入的参数和关键字参数并返回结果
        return convert_coco_poly_to_mask(*args, **kwargs)

    # 从 DETR 模型的处理图像部分复制而来，准备 COCO 检测的数据集注释
    def prepare_coco_detection(self, *args, **kwargs):
        # 发出警告日志，指出方法 `prepare_coco_detection` 将在 v4.33 版本中删除
        logger.warning_once("The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ")
        # 调用 `prepare_coco_detection_annotation` 函数处理传入的参数和关键字参数并返回结果
        return prepare_coco_detection_annotation(*args, **kwargs)

    # 从 DETR 模型的处理图像部分复制而来，准备 COCO Panoptic 的数据集注释
    def prepare_coco_panoptic(self, *args, **kwargs):
        # 发出警告日志，指出方法 `prepare_coco_panoptic` 将在 v4.33 版本中删除
        logger.warning_once("The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ")
        # 调用 `prepare_coco_panoptic_annotation` 函数处理传入的参数和关键字参数并返回结果
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    # 从 DETR 模型的处理图像部分复制而来，调整图像大小的函数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 调整图像大小到指定尺寸。尺寸可以是 `min_size`（标量）或 `(height, width)` 元组。如果尺寸是整数，则图像的较小边将匹配到该数字。
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
        # 如果 `kwargs` 中包含 `max_size` 参数，则发出警告并将其移除，建议在 `size['longest_edge']` 中指定
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 使用 `get_size_dict` 函数获取调整后的大小字典，支持 `default_to_square` 参数
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 根据大小字典中的内容调整图像大小
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            # 如果大小字典不包含所需的键，引发值错误异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 使用 `resize` 函数调整图像大小，并返回调整后的图像
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        
        # 返回调整后的图像
        return image

    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation 复制而来
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
        # 调用 `resize_annotation` 函数，将标注调整为与调整后图像匹配的大小
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)
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
        # 调用外部函数 `rescale` 对输入图像进行按比例缩放处理，并返回处理后的图像
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize_annotation
    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        # 调用外部函数 `normalize_annotation` 对给定的注释信息进行标准化处理，并返回处理后的注释信息
        return normalize_annotation(annotation, image_size=image_size)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._update_annotation_for_padded_image
    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ) -> Dict:
        """
        Update the annotation to reflect padding changes in the input image.

        Args:
            annotation (`Dict`):
                The original annotation to update.
            input_image_size (`Tuple[int, int]`):
                Size of the original input image before padding.
            output_image_size (`Tuple[int, int]`):
                Size of the padded output image after padding.
            padding:
                Details of padding applied to the image.
            update_bboxes:
                Boolean flag indicating whether to update bounding boxes in the annotation.

        Returns:
            `Dict`: Updated annotation reflecting changes due to padding.
        """
        # 调用外部函数 `_update_annotation_for_padded_image` 对给定的注释信息进行填充图像的更新处理，并返回更新后的注释信息
        return _update_annotation_for_padded_image(
            annotation, input_image_size, output_image_size, padding, update_bboxes
        )
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        # 创建一个新的注释字典
        new_annotation = {}
        # 将输出图像大小添加到新注释字典中
        new_annotation["size"] = output_image_size

        # 遍历传入的注释字典
        for key, value in annotation.items():
            if key == "masks":
                # 如果是 masks 键，获取 masks 数据并进行填充操作
                masks = value
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                masks = safe_squeeze(masks, 1)
                # 将填充后的 masks 数据添加到新注释字典中
                new_annotation["masks"] = masks
            elif key == "boxes" and update_bboxes:
                # 如果是 boxes 键且需要更新边界框
                boxes = value
                # 根据输入输出图像大小的比例，更新边界框数据
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                # 将更新后的边界框数据添加到新注释字典中
                new_annotation["boxes"] = boxes
            elif key == "size":
                # 如果是 size 键，更新输出图像大小
                new_annotation["size"] = output_image_size
            else:
                # 对于其它键直接复制到新注释字典中
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

        # 计算需要填充的底部和右侧的像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充元组
        padding = ((0, pad_bottom), (0, pad_right))
        # 对图像进行填充操作，使用指定的常数值填充
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        # 如果有注释数据，更新注释以适应填充后的图像
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        # 返回填充后的图像和更新后的注释数据
        return padded_image, annotation
    # 定义一个实例方法 `pad`，用于在图像周围填充像素值，以使它们具有相同的尺寸
    def pad(
        self,
        images: List[np.ndarray],  # 输入参数：图像列表，每个元素是一个 NumPy 数组
        annotations: Optional[List[Dict[str, Any]]] = None,  # 可选参数：注释列表，每个注释是一个字典
        constant_values: Union[float, Iterable[float]] = 0,  # 填充的像素值，可以是单个浮点数或可迭代对象
        return_pixel_mask: bool = False,  # 是否返回像素掩码
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的数据类型，可以是字符串或张量类型
        data_format: Optional[ChannelDimension] = None,  # 数据格式，通道维度的顺序
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式
        update_bboxes: bool = True,  # 是否更新边界框信息
    # 预处理图像和注释的实例方法 `preprocess`，用于执行图像的各种预处理操作
    def preprocess(
        self,
        images: ImageInput,  # 输入参数：图像，可以是单个图像或图像列表
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,  # 可选参数：注释，可以是单个注释或注释列表
        return_segmentation_masks: bool = None,  # 是否返回分割掩码
        masks_path: Optional[Union[str, pathlib.Path]] = None,  # 掩码路径
        do_resize: Optional[bool] = None,  # 是否调整大小
        size: Optional[Dict[str, int]] = None,  # 调整大小的目标尺寸
        resample=None,  # PIL 图像重采样方法
        do_rescale: Optional[bool] = None,  # 是否重新缩放
        rescale_factor: Optional[Union[int, float]] = None,  # 重新缩放因子
        do_normalize: Optional[bool] = None,  # 是否进行归一化
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差
        do_convert_annotations: Optional[bool] = None,  # 是否转换注释
        do_pad: Optional[bool] = None,  # 是否填充图像
        format: Optional[Union[str, AnnotationFormat]] = None,  # 注释格式
        return_tensors: Optional[Union[TensorType, str]] = None,  # 返回的数据类型，可以是张量或字符串
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,  # 数据格式，通道维度的顺序
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式
        **kwargs,  # 其他关键字参数
    # 后处理方法 - TODO: 添加对其他框架的支持
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process 复制，并将 Detr 替换为 Yolos
    # 将原始输出转换为最终的边界框坐标格式（top_left_x, top_left_y, bottom_right_x, bottom_right_y）。
    # 仅支持 PyTorch。

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

        # 发出警告，表明函数即将移除，建议使用替代函数 `post_process_object_detection`，并设置 `threshold=0.` 以获得相同的结果
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        # 从模型输出中获取分类置信度和预测边界框
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        # 检查输出的 logits 数量与目标尺寸数量是否一致
        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查目标尺寸的形状是否为 (batch_size, 2)
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对分类置信度进行 softmax 处理，得到每个类别的概率分布，并提取最大概率对应的类别索引和置信度分数
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将预测的边界框格式转换为 [x0, y0, x1, y1] 格式（左上角和右下角坐标）
        boxes = center_to_corners_format(out_bbox)

        # 将相对坐标 [0, 1] 转换为绝对坐标 [0, height]，根据目标尺寸缩放边界框
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 将结果组织成字典的列表，每个字典包含模型预测的分数、类别和边界框
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results

    # 从 `transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_object_detection` 复制而来，
    # 将函数名及相关说明中的 `Detr` 替换为 `Yolos`
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
        """
        Converts the raw output of [`YolosForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`YolosObjectDetectionOutput`]):
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

        # Check if target_sizes is provided and validate dimensions
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Compute probabilities and extract scores and labels
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert bounding boxes to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert relative [0, 1] coordinates to absolute [0, height] coordinates if target_sizes is provided
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Filter predictions based on score threshold and organize results into dictionaries
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
```