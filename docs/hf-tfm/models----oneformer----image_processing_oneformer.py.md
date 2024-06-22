# `.\transformers\models\oneformer\image_processing_oneformer.py`

```py
# 设置编码格式为 UTF-8

# 版权声明和许可信息

# 导入所需的库和模块
import json  # 导入用于 JSON 操作的模块
import os  # 导入用于处理文件路径的模块
import warnings  # 导入警告模块，用于发出警告消息
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库，用于处理数组
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from huggingface_hub.utils import RepositoryNotFoundError  # 导入异常类，表示存储库未找到

# 导入图像处理工具模块和相关函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  
# 导入图像处理工具相关函数和类
from ...image_transforms import (
    PaddingMode,  # 导入填充模式枚举类
    get_resize_output_image_size,  # 导入计算调整图像大小后的输出尺寸的函数
    pad,  # 导入填充图像函数
    rescale,  # 导入图像重新缩放函数
    resize,  # 导入调整图像大小函数
    to_channel_dimension_format,  # 导入将通道维度格式转换为指定格式的函数
)
# 导入图像处理工具函数
from ...image_utils import (
    ChannelDimension,  # 导入通道维度枚举类
    ImageInput,  # 导入图像输入类
    PILImageResampling,  # 导入 PIL 图像重采样枚举类
    get_image_size,  # 导入获取图像尺寸函数
    infer_channel_dimension_format,  # 导入推断图像通道维度格式的函数
    is_scaled_image,  # 导入检查图像是否已缩放的函数
    make_list_of_images,  # 导入创建图像列表函数
    to_numpy_array,  # 导入将图像转换为 NumPy 数组函数
    valid_images,  # 导入验证图像函数
)
# 导入实用工具函数和常量
from ...utils import (
    IMAGENET_DEFAULT_MEAN,  # 导入图像预处理默认均值常量
    IMAGENET_DEFAULT_STD,  # 导入图像预处理默认标准差常量
    TensorType,  # 导入张量类型枚举类
    is_torch_available,  # 导入检查是否可用 PyTorch 库的函数
    is_torch_tensor,  # 导入检查对象是否为 PyTorch 张量的函数
    logging,  # 导入日志记录模块
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 PyTorch 可用，则导入 PyTorch 库和相关模块
if is_torch_available():
    import torch  # 导入 PyTorch 库
    from torch import nn  # 导入神经网络模块
    # 创建图像的像素掩码，其中1表示有效像素，0表示填充像素

    # 参数:
    #     image (`np.ndarray`):
    #         要创建像素掩码的图像。
    #     output_size (`Tuple[int, int]`):
    #         掩码的输出大小。

    input_height, input_width = get_image_size(image, channel_dim=input_data_format)  # 获取图像的高度和宽度
    mask = np.zeros(output_size, dtype=np.int64)  # 创建一个值全为0的与输出大小相同的掩码
    mask[:input_height, :input_width] = 1  # 将掩码中对应图像区域的值设为1，表示有效像素
    return mask  # 返回生成的像素掩码
# 将给定的二进制掩码转换为游程编码(RLE)格式
def binary_mask_to_rle(mask):
    """
    将给定的二进制掩码张量(shape为(height, width))转换为游程编码(RLE)格式。
    
    参数:
        mask (`torch.Tensor` 或 `numpy.array`):
            一个二进制掩码张量,shape为(height, width),0表示背景,1表示目标segment_id或class_id。
    返回:
        `List`: 二进制掩码的游程编码列表。参考COCO API获取更多关于RLE格式的信息。
    """
    # 如果输入是 torch.Tensor,转换为numpy.array
    if is_torch_tensor(mask):
        mask = mask.numpy()
    
    # 将掩码展平成一维
    pixels = mask.flatten()
    # 在开头和结尾添加0,方便后续计算
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到像素值发生变化的位置,计算其运行长度
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    # 返回游程编码列表
    return list(runs)


# 将给定的分割图像转换为游程编码(RLE)格式
def convert_segmentation_to_rle(segmentation):
    """
    将给定的分割图像(shape为(height, width))转换为游程编码(RLE)格式。
    
    参数:
        segmentation (`torch.Tensor` 或 `numpy.array`):
            一个分割图像,shape为(height, width),每个值表示一个segment或class id。
    返回:
        `List[List]`: 一个列表的列表,每个内层列表是一个segment/class id的游程编码。
    """
    # 找到分割图像中所有唯一的segment/class id
    segment_ids = torch.unique(segmentation)
    
    # 遍历每个segment/class id,计算其游程编码并添加到结果列表
    run_length_encodings = []
    for idx in segment_ids:
        # 为当前segment/class id创建一个二进制掩码
        mask = torch.where(segmentation == idx, 1, 0)
        # 将该二进制掩码转换为游程编码
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)
    
    return run_length_encodings


# 移除低概率和无目标的掩码区域
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    使用`object_mask_threshold`二值化给定的掩码,并返回相关的`masks`、`scores`和`labels`。
    
    参数:
        masks (`torch.Tensor`):
            一个shape为`(num_queries, height, width)`的张量。
        scores (`torch.Tensor`):
            一个shape为`(num_queries)`的张量。
        labels (`torch.Tensor`):
            一个shape为`(num_queries)`的张量。
        object_mask_threshold (`float`):
            一个在0到1之间的数字,用于二值化掩码。
    异常:
        `ValueError`: 当输入张量的第一个维度不匹配时会抛出该异常。
    返回:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: 去除掉`object_mask_threshold`以下区域的`masks`、`scores`和`labels`。
    """
    # 检查输入张量的第一个维度是否匹配
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")
    
    # 计算需要保留的掩码区域
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)
    
    # 返回经过过滤的掩码、得分和标签
    return masks[to_keep], scores[to_keep], labels[to_keep]
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与类别 k 相关联的掩码
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # 计算查询类别 k 中所有内容的面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # 消除断开的小段
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 从transformers.models.detr.image_processing_detr中复制
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

    # 根据预测分数加权每个掩码
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别的实例
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查掩码是否存在且足够大以成为段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象段添加到最终分段地图中
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


# 从transformers.models.maskformer.image_processing_maskformer中复制
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
```  
    # 创建一个可选的字典，用于将实例 ID 映射到语义 ID
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    # 创建一个可选的整数，用于指定被忽略的索引
    ignore_index: Optional[int] = None,
    # 创建一个布尔类型的变量，用于指示是否减少标签数
    reduce_labels: bool = False,
def prepare_metadata(class_info):
    # 创建一个空的元数据字典
    metadata = {}
    # 创建一个空的类名列表
    class_names = []
    # 创建一个空的物体ID列表
    thing_ids = []
    # 遍历类信息字典
    for key, info in class_info.items():
        # 将类ID和对应的类名添加到元数据字典中
        metadata[key] = info["name"]
        # 将类名添加到类名列表中
        class_names.append(info["name"])
        # 如果是可视物体，将其ID添加到物体ID列表中
        if info["isthing"]:
            thing_ids.append(int(key))
    # 将物体ID列表和类名列表添加到元数据字典中
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    # 返回元数据字典
    return metadata


def load_metadata(repo_id, class_info_file):
    # 根据ID和类信息文件名创建文件路径
    fname = os.path.join("" if repo_id is None else repo_id, class_info_file)


):
    # 如果要减少标签并且未提供忽略索引，则引发值错误
    if reduce_labels and ignore_index is None:
        raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

    # 如果要减少标签，则将分割图中的0替换为忽略索引，将其他值减1
    if reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # 获取分割图中的唯一ID（基于输入的类或实例ID）
    all_labels = np.unique(segmentation_map)

    # 如果有忽略索引，则删除背景标签
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # 为每个对象实例生成二进制掩模
    binary_masks = [(segmentation_map == i) for i in all_labels]
    binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # 将实例ID转换为类ID
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
            labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
    else:
        labels = all_labels

    return binary_masks.astype(np.float32), labels.astype(np.int64)


def get_oneformer_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    max_size: Optional[int] = None,
    default_to_square: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple:
    # 计算给定大小的输出大小
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )
    # 返回输出大小
    return output_size
    # 检查文件是否存在并且是一个文件
    if not os.path.exists(fname) or not os.path.isfile(fname):
        # 如果未找到文件且 repo_id 未定义，则引发 ValueError
        if repo_id is None:
            raise ValueError(f"Could not find {fname} locally. repo_id must be defined if loading from the hub")
        
        # 尝试从数据集中下载数据以确保向后兼容性
        try:
            fname = hf_hub_download(repo_id, class_info_file, repo_type="dataset")
        except RepositoryNotFoundError:
            fname = hf_hub_download(repo_id, class_info_file)

    # 打开指定文件，以只读模式读取文件内容，将其解析为 JSON 格式
    with open(fname, "r") as f:
        class_info = json.load(f)

    # 返回解析后的类信息数据
    return class_info
# 定义一个名为 OneFormerImageProcessor 的类，继承自 BaseImageProcessor
class OneFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a OneFormer image processor. The image processor can be used to prepare image(s), task input(s) and
    optional text inputs and targets for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    """

    # 定义 model_input_names 属性为包含字符串 "pixel_values", "pixel_mask", "task_inputs" 的列表
    model_input_names = ["pixel_values", "pixel_mask", "task_inputs"]

    # 初始化方法，接受一系列参数来配置图像处理器的行为
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
        repo_path: Optional[str] = "shi-labs/oneformer_demo",
        class_info_file: str = None,
        num_text: Optional[int] = None,
        **kwargs,
    ):
        # 如果 kwargs 中存在 "max_size" 键，将其赋值给 self._max_size
        if "max_size" in kwargs:
            self._max_size = kwargs.pop("max_size")
        else:
            self._max_size = 1333

        # 根据传入的 size 字典或默认大小设置 size 参数
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        # 如果 kwargs 中存在 "reduce_labels" 键，发出警告，使用 do_reduce_labels 替代
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` argument is deprecated and will be removed in v4.27. "
                "Please use `do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        # 如果未提供 class_info_file 参数，则抛出数值错误
        if class_info_file is None:
            raise ValueError("You must provide a `class_info_file`")

        # 调用父类的初始化方法
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.ignore_index = ignore_index
        self.do_reduce_labels = do_reduce_labels
        self.class_info_file = class_info_file
        self.repo_path = repo_path
        # 准备元数据，加载元数据到 metadata 属性
        self.metadata = prepare_metadata(load_metadata(repo_path, class_info_file))
        self.num_text = num_text

    # 对输入图像进行调整大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```py 
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        # 检查是否传入了 'max_size' 参数，如果是，则发出警告，建议使用 size['longest_edge'] 替代
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            # 将 max_size 参数弹出并赋值给 max_size 变量
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        # 调用 get_size_dict 函数获取处理后的尺寸信息
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        # 检查 size 是否包含 'shortest_edge' 和 'longest_edge' 键
        if "shortest_edge" in size and "longest_edge" in size:
            # 如果包含，则将 'shortest_edge' 和 'longest_edge' 键的值赋给 size 和 max_size 变量
            size, max_size = size["shortest_edge"], size["longest_edge"]
        # 检查 size 是否包含 'height' 和 'width' 键
        elif "height" in size and "width" in size:
            # 如果包含，则将 'height' 和 'width' 键的值组成元组赋给 size 变量，并将 max_size 变量置为 None
            size = (size["height"], size["width"])
            max_size = None
        else:
            # 如果 size 不符合以上条件，则抛出 ValueError 异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        # 调用 get_oneformer_resize_output_image_size 函数获取调整后的图像尺寸
        size = get_oneformer_resize_output_image_size(
            image=image, size=size, max_size=max_size, default_to_square=False, input_data_format=input_data_format
        )
        # 调用 resize 函数对图像进行调整大小操作
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format
        )
        # 返回调整大小后的图像
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    # 缩放图像到给定的比例
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个函数，用于对图像进行重新缩放，返回缩放后的图像数据
    def rescale_image(image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]] = None, input_data_format: Optional[Union[str, ChannelDimension]] = None) -> np.ndarray:
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

    # 从transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor中复制方法convert_segmentation_map_to_binary_masks
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        # 如果reduce_labels不为空，则使用reduce_labels；否则使用self.reduce_labels
        reduce_labels = reduce_labels if reduce_labels is not None else self.reduce_labels
        # 如果ignore_index不为空，则使用ignore_index；否则使用self.ignore_index
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        # 调用convert_segmentation_map_to_binary_masks方法，返回二进制掩码结果
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            reduce_labels=reduce_labels,
        )

    # 定义一个函数，用于预处理图像数据和分割图，返回BatchFeature对象
    def __call__(self, images, task_inputs=None, segmentation_maps=None, **kwargs) -> BatchFeature:
        return self.preprocess(images, task_inputs=task_inputs, segmentation_maps=segmentation_maps, **kwargs)

    # 定义一个私有的预处理方法，用于对图像进行调整大小、重新缩放和正则化处理
    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 此处缺少方法体，需要补充具体流程
    ):  
        # 如果需要对图像进行调整大小
        if do_resize:
            # 对图像进行调整大小操作
            image = self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
        # 如果需要对图像进行重新缩放
        if do_rescale:
            # 对图像进行重新缩放操作
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        # 如果需要对图像进行归一化
        if do_normalize:
            # 对图像进行归一化操作
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # 返回处理后的图像
        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 所有的转换都期望输入为 numpy 数组
        image = to_numpy_array(image)
        # 如果图像已经进行过缩放操作，并且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 推断通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # 对图像进行预处理
        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        # 如果有指定数据格式，则将图像转换为该格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回处理后的图像
        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        # 将分割地图转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果分割地图的维度为2，则添加通道维度，某些转换需要这个维度
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果输入数据格式未指定，则推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # TODO: (Amy)
        # 重新设计分割地图处理流程，包括减少标签和调整大小，不丢弃 > 255 的段ID
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )
        # 如果为了处理而添加了额外的通道维度，则去除它
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        # 返回预处理后的分割地图
        return segmentation_map

    def preprocess(
        self,
        images: ImageInput,
        task_inputs: Optional[List[str]] = None,
        segmentation_maps: Optional[ImageInput] = None,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 从 transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image 复制而来
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def pad(
        self,
        images: List[np.ndarray],  # 输入参数 images 是一个 numpy 数组的列表
        constant_values: Union[float, Iterable[float]] = 0,  # 指定填充值，可以是单个值或者可迭代对象
        return_pixel_mask: bool = True,  # 是否返回像素掩码
        return_tensors: Optional[Union[str, TensorType]] = None,  # 是否返回张量，可以是字符串或张量类型的可选值
        data_format: Optional[ChannelDimension] = None,  # 数据格式，可选值包括通道维度
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可以是字符串或通道维度的可选值
    ) -> np.ndarray:  # 返回一个 numpy 数组
        """
        Pad an image with zeros to the given size.  # 使用零填充图像到指定大小
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)  # 获取图像的高度和宽度
        output_height, output_width = output_size  # 获取输出图像的高度和宽度

        pad_bottom = output_height - input_height  # 计算垂直方向所需填充大小
        pad_right = output_width - input_width  # 计算水平方向所需填充大小
        padding = ((0, pad_bottom), (0, pad_right))  # 构建填充的位置信息
        padded_image = pad(  # 对图像进行填充
            image,  # 输入图像
            padding,  # 填充位置
            mode=PaddingMode.CONSTANT,  # 指定填充模式为常量填充
            constant_values=constant_values,  # 指定常量填充的值
            data_format=data_format,  # 指定数据格式
            input_data_format=input_data_format,  # 指定输入数据格式
        )
        return padded_image  # 返回填充后的图像
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
        # Get the maximum height and width of images in the batch for padding
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # Pad each image in the batch to the specified size
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
        # Create a dictionary containing the padded images
        data = {"pixel_values": padded_images}

        # If specified, generate pixel masks for the padded images
        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # Return BatchFeature object with the padded images and pixel masks if requested
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 获取语义注释，给定标签和类别对象数量，返回注释的类别、掩码和文本描述
    def get_semantic_annotations(self, label, num_class_obj):
        # 获取标签中的类别和掩码
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        # 初始化文本描述列表
        texts = ["a semantic photo"] * self.num_text
        # 初始化类别和掩码列表
        classes = []
        masks = []

        # 遍历标签中的类别和掩码
        for idx in range(len(annotation_classes)):
            # 获取类别和掩码
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]
            # 如果掩码不全为假值
            if not np.all(mask is False):
                # 如果类别不在已记录的类别中
                if class_id not in classes:
                    # 获取类别名称
                    cls_name = self.metadata[str(class_id)]
                    # 添加类别和掩码到列表中
                    classes.append(class_id)
                    masks.append(mask)
                    # 更新对应类别对象数量
                    num_class_obj[cls_name] += 1
                else:
                    # 如果类别已经存在，更新掩码
                    idx = classes.index(class_id)
                    masks[idx] += mask
                    masks[idx] = np.clip(masks[idx], 0, 1)

        # 统计已记录类别的数量
        num = 0
        # 遍历所有类别名称
        for i, cls_name in enumerate(self.metadata["class_names"]):
            # 如果该类别对象数量大于零
            if num_class_obj[cls_name] > 0:
                # 根据对象数量添加对应文本描述
                for _ in range(num_class_obj[cls_name]):
                    # 如果文本描述数量超过预设数量，结束
                    if num >= len(texts):
                        break
                    # 添加文本描述
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        # 转换列表为数组形式并返回
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts

    # 获取实例注释，给定标签和类别对象数量，返回注释的类别、掩码和文本描述
    def get_instance_annotations(self, label, num_class_obj):
        # 获取标签中的类别和掩码
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        # 初始化文本描述列表
        texts = ["an instance photo"] * self.num_text
        # 初始化类别和掩码列表
        classes = []
        masks = []

        # 遍历标签中的类别和掩码
        for idx in range(len(annotation_classes)):
            # 获取类别和掩码
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]

            # 如果类别属于物体类别
            if class_id in self.metadata["thing_ids"]:
                # 如果掩码不全为假值
                if not np.all(mask is False):
                    # 获取类别名称
                    cls_name = self.metadata[str(class_id)]
                    # 添加类别和掩码到列表中
                    classes.append(class_id)
                    masks.append(mask)
                    # 更新对应类别对象数量
                    num_class_obj[cls_name] += 1

        # 统计已记录类别的数量
        num = 0
        # 遍历所有类别名称
        for i, cls_name in enumerate(self.metadata["class_names"]):
            # 如果该类别对象数量大于零
            if num_class_obj[cls_name] > 0:
                # 根据对象数量添加对应文本描述
                for _ in range(num_class_obj[cls_name]):
                    # 如果文本描述数量超过预设数量，结束
                    if num >= len(texts):
                        break
                    # 添加文本描述
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        # 转换列表为数组形式并返回
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts
    # 获取全景分割注释
    def get_panoptic_annotations(self, label, num_class_obj):
        # 从注释数据中获取类别和掩码
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]
    
        # 创建预定义的文本描述列表
        texts = ["an panoptic photo"] * self.num_text
        classes = []
        masks = []
    
        # 遍历每个注释对象
        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx].data
            # 如果掩码不为全False
            if not np.all(mask is False):
                cls_name = self.metadata[str(class_id)]
                classes.append(class_id)
                masks.append(mask)
                # 更新类别对象数量
                num_class_obj[cls_name] += 1
    
        num = 0
        # 遍历每个类别
        for i, cls_name in enumerate(self.metadata["class_names"]):
            # 如果该类别有对应的对象
            if num_class_obj[cls_name] > 0:
                # 添加对应数量的文本描述
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1
    
        # 将classes和masks转换为numpy数组
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts
    
    # 编码输入数据
    def encode_inputs(
        self,
        pixel_values_list: List[ImageInput],
        task_inputs: List[str],
        segmentation_maps: ImageInput = None,
        instance_id_to_semantic_id: Optional[Union[List[Dict[int, int]], Dict[int, int]]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 此函数用于对输入数据进行编码处理
    
    # 后处理语义分割输出
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ):
        # 此函数用于对语义分割的输出进行后处理
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

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
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
    # 从 transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation 复制而来的方法
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,  # 阈值，用于过滤语义分割预测中的置信度低于该值的部分
        mask_threshold: float = 0.5,  # 阈值，用于二值化实例分割掩码
        overlap_mask_area_threshold: float = 0.8,  # 阈值，用于确定是否应将两个实例的掩码合并为一个
        label_ids_to_fuse: Optional[Set[int]] = None,  # 用于指定应合并的标签ID的集合，如果为None，则合并所有标签
        target_sizes: Optional[List[Tuple[int, int]]] = None,  # 目标尺寸的列表，用于调整掩码的大小
```