# `.\models\oneformer\image_processing_oneformer.py`

```py
# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""
Image processor class for OneFormer.
"""

import json  # 导入处理 JSON 的模块
import os  # 导入操作系统路径的模块
import warnings  # 导入警告模块
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 数学库
from huggingface_hub import hf_hub_download  # 导入从 HF Hub 下载资源的函数
from huggingface_hub.utils import RepositoryNotFoundError  # 导入 HF Hub 中的错误类

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理相关的工具函数和类
from ...image_transforms import (  # 导入图像转换相关函数和枚举
    PaddingMode,
    get_resize_output_image_size,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (  # 导入图像处理工具函数
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import (  # 导入通用工具函数和常量
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

if is_torch_available():
    import torch  # 导入 PyTorch 库
    from torch import nn  # 导入神经网络模块

# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]  # 返回每个值中的最大值的列表

# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])  # 推断输入数据的通道维度格式

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])  # 获取通道维度为第一维时的最大高度和宽度
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])  # 获取通道维度为最后一维时的最大高度和宽度
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")  # 如果通道维度格式无效，则抛出错误
    return (max_height, max_width)  # 返回最大高度和宽度的列表

# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    # 参数说明：
    # image: np.ndarray，表示输入的图像数据，是一个 NumPy 数组
    # output_size: Tuple[int, int]，表示期望输出的图像尺寸，以元组形式给出，包含两个整数值
    # input_data_format: Optional[Union[str, ChannelDimension]] = None，表示输入数据的格式，可以是字符串或者 ChannelDimension 类型的可选值，默认为 None
# 创建一个像素掩码，其中 1 表示有效像素，0 表示填充像素
def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    # 获取图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建一个全零的像素掩码数组，指定数据类型为 np.int64
    mask = np.zeros(output_size, dtype=np.int64)
    # 将掩码的有效区域（根据图像尺寸）设置为 1
    mask[:input_height, :input_width] = 1
    return mask


# 从 transformers.models.detr.image_processing_detr.binary_mask_to_rle 复制而来
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
    # 如果输入的 mask 是 torch.Tensor，则转换为 numpy.array
    if is_torch_tensor(mask):
        mask = mask.numpy()

    # 将二进制掩码展平为一维数组
    pixels = mask.flatten()
    # 在数组两端各添加一个零，便于计算 run-length 编码
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到不同值的起始位置
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 计算 run-length 编码的长度
    runs[1::2] -= runs[::2]
    return list(runs)


# 从 transformers.models.detr.image_processing_detr.convert_segmentation_to_rle 复制而来
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    # 获取唯一的分割标识符（segment_ids）
    segment_ids = torch.unique(segmentation)

    # 存储所有分割标识符的 run-length 编码
    run_length_encodings = []
    # 对每个分割标识符进行处理
    for idx in segment_ids:
        # 创建一个二进制掩码，表示当前分割标识符的区域
        mask = torch.where(segmentation == idx, 1, 0)
        # 将二进制掩码转换为 run-length 编码
        rle = binary_mask_to_rle(mask)
        # 将当前分割标识符的 run-length 编码添加到结果列表中
        run_length_encodings.append(rle)

    return run_length_encodings


# 从 transformers.models.detr.image_processing_detr.remove_low_and_no_objects 复制而来
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
    """
    # 检查 `masks`、`scores` 和 `labels` 的形状是否一致，若不一致则抛出数值错误异常
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    # 创建布尔索引 `to_keep`，用于选择除了 `num_labels` 以外的标签，并且对应的分数大于 `object_mask_threshold`
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    # 根据布尔索引 `to_keep` 过滤出符合条件的 `masks`、`scores` 和 `labels`
    return masks[to_keep], scores[to_keep], labels[to_keep]
# 从给定的掩码和概率中检查第 k 类的段是否有效
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # 获取与第 k 类相关联的掩码
    mask_k = mask_labels == k
    # 计算第 k 类掩码的总面积
    mask_k_area = mask_k.sum()

    # 计算预测为第 k 类的所有掩码的总面积
    original_area = (mask_probs[k] >= mask_threshold).sum()
    # 检查掩码是否存在且面积足够大
    mask_exists = mask_k_area > 0 and original_area > 0

    # 如果掩码存在，则进一步检查是否与其他掩码有重叠
    if mask_exists:
        # 计算实际掩码区域与预测掩码区域的比率
        area_ratio = mask_k_area / original_area
        # 如果重叠面积比率不满足阈值要求，则认为掩码不存在
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# 计算分割结果的段和分割信息
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    # 根据目标大小或原始大小设置图像高度和宽度
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    # 创建一个全零的分割图像
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    # 初始化空的段列表
    segments: List[Dict] = []

    # 如果指定了目标大小，则插值调整掩码的大小
    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # 将每个掩码按其预测分数加权
    mask_probs *= pred_scores.view(-1, 1, 1)
    # 确定每个像素点属于哪个类别的掩码
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # 跟踪每个类别实例的数量
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # 检查掩码是否存在且足够大来作为一个段
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # 将当前对象的段添加到最终分割图中
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
            # 如果应该融合，则更新类别实例的内存
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    ignore_index: Optional[int] = None,
    reduce_labels: bool = False,
):
    if reduce_labels and ignore_index is None:
        raise ValueError("If `reduce_labels` is True, `ignore_index` must be provided.")

    if reduce_labels:
        # 将标签值减少并处理忽略索引
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # 获取唯一的标签ID（基于输入的类别或实例ID）
    all_labels = np.unique(segmentation_map)

    # 如果存在忽略索引，则去除背景标签
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # 为每个对象实例生成二进制掩码
    binary_masks = [(segmentation_map == i) for i in all_labels]
    binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # 如果需要，将实例ID转换为类别ID
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            # 获取类别ID
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
    """
    根据所需的大小计算输出大小。

    Args:
        image (`np.ndarray`):
            输入图像。
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            输出图像的大小。
        max_size (`int`, *optional*):
            输出图像的最大大小。
        default_to_square (`bool`, *optional*, 默认为 `True`):
            如果未提供大小，是否默认为正方形。
        input_data_format (`ChannelDimension` or `str`, *optional*):
            输入图像的通道维度格式。如果未设置，则使用输入的推断格式。

    Returns:
        `Tuple[int, int]`: 输出大小。
    """
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )
    return output_size


def prepare_metadata(class_info):
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        # 添加类别名称到元数据字典中
        metadata[key] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            # 如果是物体类别，则将其ID添加到列表中
            thing_ids.append(int(key))
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata
# 根据给定的 repo_id 和 class_info_file 构建元数据文件路径
def load_metadata(repo_id, class_info_file):
    fname = os.path.join("" if repo_id is None else repo_id, class_info_file)

    # 检查文件是否存在且是一个普通文件
    if not os.path.exists(fname) or not os.path.isfile(fname):
        # 如果 repo_id 未定义且文件不存在，则抛出值错误
        if repo_id is None:
            raise ValueError(f"Could not find {fname} locally. repo_id must be defined if loading from the hub")
        # 尝试从数据集下载文件以保持向后兼容性
        try:
            fname = hf_hub_download(repo_id, class_info_file, repo_type="dataset")
        except RepositoryNotFoundError:
            # 如果下载失败，则再次尝试下载
            fname = hf_hub_download(repo_id, class_info_file)

    # 打开元数据文件并加载为 JSON 格式
    with open(fname, "r") as f:
        class_info = json.load(f)

    # 返回加载的类信息
    return class_info


class OneFormerImageProcessor(BaseImageProcessor):
    r"""
    构造一个 OneFormer 图像处理器。该图像处理器用于准备图像、任务输入及可选的文本输入和目标，以供模型使用。

    此图像处理器继承自 `BaseImageProcessor`，其中包含大多数主要方法。用户应参考该超类以获取关于这些方法的更多信息。
    """

    # 模型输入名称列表
    model_input_names = ["pixel_values", "pixel_mask", "task_inputs"]

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
        # 检查 kwargs 中是否有 "max_size" 参数，如果有则将其取出赋值给 self._max_size，否则默认为 1333
        if "max_size" in kwargs:
            self._max_size = kwargs.pop("max_size")
        else:
            self._max_size = 1333

        # 根据传入的 size 参数确定图片的尺寸，如果 size 为 None，则设定默认的最短边为 800，最长边为 self._max_size
        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        # 调用 get_size_dict 函数，根据传入参数获取最终的尺寸字典，max_size 参数为 self._max_size，默认不将图像调整为正方形
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        # 如果 kwargs 中包含 "reduce_labels" 参数，则发出警告并将其取出赋值给 do_reduce_labels
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` argument is deprecated and will be removed in v4.27. "
                "Please use `do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        # 如果 class_info_file 参数为 None，则抛出 ValueError 异常，提示需要提供 class_info_file 参数
        if class_info_file is None:
            raise ValueError("You must provide a `class_info_file`")

        # 调用父类的初始化方法，传入所有 kwargs 参数
        super().__init__(**kwargs)
        # 将 do_resize 参数赋值给实例属性 self.do_resize
        self.do_resize = do_resize
        # 将 size 参数赋值给实例属性 self.size
        self.size = size
        # 将 resample 参数赋值给实例属性 self.resample
        self.resample = resample
        # 将 do_rescale 参数赋值给实例属性 self.do_rescale
        self.do_rescale = do_rescale
        # 将 rescale_factor 参数赋值给实例属性 self.rescale_factor
        self.rescale_factor = rescale_factor
        # 将 do_normalize 参数赋值给实例属性 self.do_normalize
        self.do_normalize = do_normalize
        # 如果 image_mean 参数不为 None，则将其赋值给 self.image_mean，否则使用默认值 IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 如果 image_std 参数不为 None，则将其赋值给 self.image_std，否则使用默认值 IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 将 ignore_index 参数赋值给实例属性 self.ignore_index
        self.ignore_index = ignore_index
        # 将 do_reduce_labels 参数赋值给实例属性 self.do_reduce_labels
        self.do_reduce_labels = do_reduce_labels
        # 将 class_info_file 参数赋值给实例属性 self.class_info_file
        self.class_info_file = class_info_file
        # 将 repo_path 参数赋值给实例属性 self.repo_path
        self.repo_path = repo_path
        # 调用 load_metadata 函数加载元数据，并通过 prepare_metadata 函数准备元数据，赋值给 self.metadata
        self.metadata = prepare_metadata(load_metadata(repo_path, class_info_file))
        # 将 num_text 参数赋值给实例属性 self.num_text
        self.num_text = num_text
        # 定义有效的处理器键列表，用于后续处理器的检查和操作
        self._valid_processor_keys = [
            "images",
            "task_inputs",
            "segmentation_maps",
            "instance_id_to_semantic_id",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "ignore_index",
            "do_reduce_labels",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 定义 resize 方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,  # 待调整大小的图像数组
        size: Dict[str, int],  # 目标尺寸字典，包含宽度和高度
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        data_format=None,  # 数据格式参数，可以不指定
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，可选
        **kwargs,  # 其他关键字参数
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        # 检查是否有 'max_size' 参数，如果有则发出警告并将其移除
        if "max_size" in kwargs:
            warnings.warn(
                "The `max_size` parameter is deprecated and will be removed in v4.27. "
                "Please specify in `size['longest_edge'] instead`.",
                FutureWarning,
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        
        # 根据给定的 size 和 max_size 获取尺寸信息字典
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        
        # 如果 size 包含 'shortest_edge' 和 'longest_edge' 键，则提取对应的尺寸和最大尺寸
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        # 如果 size 包含 'height' 和 'width' 键，则将其转换为 (height, width) 形式，并清除 max_size
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            # 如果 size 不符合预期，抛出 ValueError 异常
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        
        # 使用 get_oneformer_resize_output_image_size 函数获取调整后的图片尺寸
        size = get_oneformer_resize_output_image_size(
            image=image, size=size, max_size=max_size, default_to_square=False, input_data_format=input_data_format
        )
        
        # 调用 resize 函数对图片进行调整大小操作
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format
        )
        
        # 返回调整大小后的图片
        return image
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

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.convert_segmentation_map_to_binary_masks
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ):
        # Determine if the 'reduce_labels' parameter should use the instance variable if not explicitly provided
        reduce_labels = reduce_labels if reduce_labels is not None else self.reduce_labels
        # Determine if the 'ignore_index' parameter should use the instance variable if not explicitly provided
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        # Call the utility function to convert segmentation map to binary masks
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            reduce_labels=reduce_labels,
        )

    def __call__(self, images, task_inputs=None, segmentation_maps=None, **kwargs) -> BatchFeature:
        # Delegate the preprocessing task to the 'preprocess' method
        return self.preprocess(images, task_inputs=task_inputs, segmentation_maps=segmentation_maps, **kwargs)

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
    ):
        # 如果需要调整大小，则调用 resize 方法对图像进行处理
        if do_resize:
            image = self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
        # 如果需要重新缩放，则调用 rescale 方法对图像进行处理
        if do_rescale:
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        # 如果需要归一化，则调用 normalize 方法对图像进行处理
        if do_normalize:
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        # 返回预处理后的图像
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
        # 转换图像为 numpy 数组格式，因为后续的处理都需要 numpy 数组作为输入
        image = to_numpy_array(image)
        # 如果图像已经被缩放并且需要进行重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 推断输入数据的通道维度格式，如果未指定的话
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # 调用内部方法 _preprocess 对图像进行实际处理
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
        # 如果指定了输出数据的通道维度格式，则转换图像到指定格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回预处理后的图像数据
        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        # 将 segmentation_map 转换为 NumPy 数组，确保数据类型一致
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果 segmentation_map 的维度为2，表示缺少通道维度，需要添加通道维度以便进行某些转换操作
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]  # 在第0维度上添加一个维度
            input_data_format = ChannelDimension.FIRST  # 设置数据格式为第一种格式
        else:
            added_channel_dim = False
            # 如果未指定输入数据格式，通过推断获取通道维度的格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # TODO: (Amy)
        # 重新设计分割地图处理过程，包括减少标签和调整大小，确保不丢弃大于255的分割标识。
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )
        # 如果为了处理而添加了额外的通道维度，则去除该维度
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
    # 从 transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image 复制而来
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个函数 pad，用于对图像进行填充操作，使其达到指定的大小
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出图像的目标高度和宽度
        output_height, output_width = output_size

        # 计算需要在图像底部和右侧添加的填充像素数
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        # 构建填充元组，表示在上、下、左、右四个方向的填充像素数
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
        # 返回填充后的图像
        return padded_image

    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor.pad
    # 定义了一个名为 pad 的方法，用于在 ViltImageProcessor 类中进行图像填充操作
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
        # 获取批量图像中最大高度和宽度，并计算需要填充的大小
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对每张图像进行填充操作，返回填充后的图像列表
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
        # 构造数据字典，包含填充后的像素值列表
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 对每张图像生成相应的像素掩码，并构造掩码列表
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            # 将掩码列表加入数据字典中
            data["pixel_mask"] = masks

        # 返回批量特征对象，包含填充后的数据和指定的张量类型
        return BatchFeature(data=data, tensor_type=return_tensors)
    # 获取语义注释，返回实例类别、掩码和文本描述
    def get_semantic_annotations(self, label, num_class_obj):
        # 从标签中获取类别和掩码
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        # 初始化文本描述
        texts = ["a semantic photo"] * self.num_text
        # 用于存储类别和掩码的空列表
        classes = []
        masks = []

        # 遍历每个标注的类别和掩码
        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]
            # 如果掩码不全为假
            if not np.all(mask is False):
                # 如果类别不在已有的类别列表中，则添加该类别和对应的掩码
                if class_id not in classes:
                    cls_name = self.metadata[str(class_id)]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                else:
                    # 否则，更新已有类别的掩码并进行截断处理
                    idx = classes.index(class_id)
                    masks[idx] += mask
                    masks[idx] = np.clip(masks[idx], 0, 1)

        # 根据类别的数量，生成文本描述
        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        # 转换为 NumPy 数组
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts

    # 获取实例注释，返回实例类别、掩码和文本描述
    def get_instance_annotations(self, label, num_class_obj):
        # 从标签中获取类别和掩码
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        # 初始化文本描述
        texts = ["an instance photo"] * self.num_text
        # 用于存储类别和掩码的空列表
        classes = []
        masks = []

        # 遍历每个标注的类别和掩码
        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]

            # 如果类别在元数据的事物类别中，并且掩码不全为假
            if class_id in self.metadata["thing_ids"]:
                if not np.all(mask is False):
                    cls_name = self.metadata[str(class_id)]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1

        # 根据类别的数量，生成文本描述
        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        # 转换为 NumPy 数组
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts
    # 定义一个方法用于从标签中提取全景注释信息，包括类别和掩码
    def get_panoptic_annotations(self, label, num_class_obj):
        # 从标签中获取类别信息
        annotation_classes = label["classes"]
        # 从标签中获取掩码信息
        annotation_masks = label["masks"]

        # 初始化文本列表，填充为 "an panoptic photo" 的形式
        texts = ["an panoptic photo"] * self.num_text
        # 初始化类别和掩码列表
        classes = []
        masks = []

        # 遍历每个标签中的类别和掩码
        for idx in range(len(annotation_classes)):
            # 获取当前类别的ID
            class_id = annotation_classes[idx]
            # 获取当前类别对应的掩码数据
            mask = annotation_masks[idx].data
            # 如果掩码不全为 False，则处理该掩码
            if not np.all(mask is False):
                # 根据类别ID获取类别名称
                cls_name = self.metadata[str(class_id)]
                # 将类别ID和掩码添加到列表中
                classes.append(class_id)
                masks.append(mask)
                # 更新该类别在 num_class_obj 中的计数
                num_class_obj[cls_name] += 1

        # 初始化计数器 num
        num = 0
        # 遍历元数据中的类别名称
        for i, cls_name in enumerate(self.metadata["class_names"]):
            # 如果该类别在 num_class_obj 中的计数大于 0
            if num_class_obj[cls_name] > 0:
                # 根据类别计数循环更新文本列表中的内容
                for _ in range(num_class_obj[cls_name]):
                    # 如果 num 超过文本列表长度，则退出循环
                    if num >= len(texts):
                        break
                    # 更新文本列表中的内容为 "a photo with a {cls_name}"
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        # 将类别列表和掩码列表转换为 numpy 数组，并返回类别、掩码和文本列表
        classes = np.array(classes)
        masks = np.array(masks)
        return classes, masks, texts

    # 定义一个方法用于处理语义分割后的输出结果
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ):
        # 方法功能待补充
        pass
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
        # Extract class queries logits and masks queries logits from the model outputs
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Remove the null class from class queries logits
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # [batch_size, num_queries, num_classes]
        # Apply sigmoid to masks queries logits to get probabilities
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Compute semantic segmentation logits by combining class probabilities and mask probabilities
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps if target_sizes are provided
        if target_sizes is not None:
            # Check if batch size matches the length of target_sizes
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            # Iterate over batch to resize and compute semantic maps
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # Extract the semantic map by taking argmax over class dimension
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # Compute semantic segmentation maps by taking argmax over the class dimension
            semantic_segmentation = segmentation.argmax(dim=1)
            # Convert to list format for consistency
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # Return the computed semantic segmentation maps
        return semantic_segmentation
    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation
    # 从 transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation 复制而来
    
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
```