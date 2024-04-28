# `.\transformers\models\bridgetower\image_processing_bridgetower.py`

```
# 设置文件编码为 UTF-8

# 版权声明和许可证信息，指明代码的版权归属和使用许可
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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

# 导入必要的模块和函数
"""Image processor class for BridgeTower."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
# 导入 NumPy 库，用于处理数组数据
import numpy as np

# 导入必要的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
from ...image_transforms import PaddingMode, center_crop, pad, resize, to_channel_dimension_format
# 导入图像处理工具函数
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
)
# 导入日志记录工具
from ...utils import TensorType, is_vision_available, logging

# 如果视觉处理模块可用，则导入必要的 PIL 库
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义函数：从可迭代对象中获取每个索引位置的最大值并返回
# Copied from transformers.models.vilt.image_processing_vilt.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# 定义函数：为图像生成像素掩码，其中 1 表示有效像素，0 表示填充像素
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
    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    # 创建与输出尺寸相同的零矩阵
    mask = np.zeros(output_size, dtype=np.int64)
    # 将有效像素位置置为 1
    mask[:input_height, :input_width] = 1
    return mask


# 定义函数：获取批量图像中所有图像的最大高度和宽度
# Copied from transformers.models.vilt.image_processing_vilt.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    # 如果未提供输入数据格式，则推断第一个图像的格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    # 如果输入数据格式为“FIRST”，则获取批量图像中的最大高度和宽度
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    # 如果输入数据格式为通道维度在最后
    elif input_data_format == ChannelDimension.LAST:
        # 获取所有图像的形状，并找到它们的最大高度和宽度，忽略通道维度
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    # 如果输入数据格式不是通道维度在最后
    else:
        # 抛出值错误，指示通道维度格式无效
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    # 返回最大高度和最大宽度的元组
    return (max_height, max_width)
# 导入所需库
# 从指定路径导入字节流操作类 BytesIO
from io import BytesIO
# 导入 zipfile 库，用于 ZIP 文件处理
import zipfile
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 PIL 库，用于图像处理
from PIL import Image as PILImage
# 导入 PIL.Image 里面的 Resampling 方法，用于图像重采样
from PIL.Image import Resampling as PILImageResampling
# 导入 typing 库，用于类型提示
from typing import Optional, Union, Tuple, Dict, List
# 从 transformers.models.vilt.image_processing_vilt 中导入基础图像处理类 BaseImageProcessor
from transformers.models.vilt.image_processing_vilt import BaseImageProcessor
# 从 transformers.models.vilt.image_processing_vilt 中导入图像大小获取函数 get_image_size 和图像大小字典获取函数 get_size_dict
from transformers.models.vilt.image_processing_vilt import get_image_size, get_size_dict

# 定义函数 get_resize_output_image_size，用于计算调整后的图像尺寸
def get_resize_output_image_size(
    input_image: np.ndarray,  # 输入图像的 numpy 数组表示
    shorter: int = 800,  # 短边的目标尺寸，默认为 800
    longer: int = 1333,  # 长边的目标尺寸，默认为 1333
    size_divisor: int = 32,  # 尺寸除数，默认为 32
    input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
) -> Tuple[int, int]:  # 返回值为元组，表示调整后的高度和宽度
    # 获取输入图像的高度和宽度
    input_height, input_width = get_image_size(input_image, input_data_format)
    # 确定最小和最大尺寸
    min_size, max_size = shorter, longer
    # 计算缩放比例
    scale = min_size / min(input_height, input_width)
    # 根据输入图像的高宽比例确定调整后的高度和宽度
    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size
    # 如果调整后的高度或宽度超过了最大尺寸，则重新计算缩放比例
    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width
    # 对调整后的高度和宽度进行取整，并按照 size_divisor 进行调整
    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor
    # 返回调整后的高度和宽度
    return new_height, new_width

# 定义类 BridgeTowerImageProcessor，用于构建 BridgeTower 图像处理器
class BridgeTowerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BridgeTower image processor.
    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]

    # 初始化函数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整，默认为 True
        size: Dict[str, int] = 288,  # 调整后图像的大小，默认为 288
        size_divisor: int = 32,  # 尺寸除数，默认为 32
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为 BICUBIC
        do_rescale: bool = True,  # 是否进行重新缩放，默认为 True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为 1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为 True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为 OPENAI_CLIP_MEAN
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为 OPENAI_CLIP_STD
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为 True
        do_pad: bool = True,  # 是否进行填充，默认为 True
        **kwargs,  # 其他参数
    ) -> None:  # 无返回值
        # 如果 kwargs 中包含 "pad_and_return_pixel_mask" 参数，则将其作为 do_pad 的值
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将 size 转换为字典格式，若为 None 则使用默认值
        size = size if size is not None else {"shortest_edge": 288}
        size = get_size_dict(size, default_to_square=False)
        # 初始化各种参数
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_center_crop = do_center_crop

    # 定义 ViltImageProcessor 类中的 resize 方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Resizes the shorter side of the image to `size["shortest_edge"]` while preserving the aspect ratio. If the
        longer side is larger than the max size `(int(`size["shortest_edge"]` * 1333 / 800))`, the longer side is then
        resized to the max size while preserving the aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Controls the size of the output image. Should be of the form `{"shortest_edge": int}`.
            size_divisor (`int`, defaults to 32):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling` filter, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 将 size 参数转换为标准格式
        size = get_size_dict(size, default_to_square=False)
        # 如果 size 字典中不包含 "shortest_edge" 键，则抛出 ValueError 异常
        if "shortest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        # 获取最短边的长度
        shorter = size["shortest_edge"]
        # 计算最长边的长度，使得长宽比保持不变，且最长边不超过 `(int(size["shortest_edge"] * 1333 / 800))`
        longer = int(1333 / 800 * shorter)
        # 获取调整后的输出图像大小
        output_size = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor, input_data_format=input_data_format
        )
        # 调整图像大小并返回结果
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image in the form `{"height": h, "width": w}`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        output_size = size["shortest_edge"]
        # 返回按指定大小中心裁剪的图像，如果输入尺寸沿任一边小于 `crop_size`，则图像将填充0，然后进行中心裁剪
        return center_crop(
            image,
            size=(output_size, output_size),  # 裁剪的输出尺寸为 (output_size, output_size)
            data_format=data_format,  # 图像的通道维度格式
            input_data_format=input_data_format,  # 输入图像的通道维度格式
            **kwargs,  # 其它关键字参数
        )

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
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
        output_height, output_width = output_size  # 获取输出图像的高度和宽度

        # 计算需要填充的底部和右侧的大小
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))  # 构造填充元组
        # 使用常数值填充图像，以达到给定的尺寸
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,  # 使用常数填充模式
            constant_values=constant_values,  # 填充的常数值
            data_format=data_format,  # 图像的通道维度格式
            input_data_format=input_data_format,  # 输入图像的通道维度格式
        )
        return padded_image  # 返回填充后的图像

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad
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
        # 计算需要填充的大小，以保证批次中最大的高度和宽度一致
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        # 对批次中的每张图像进行填充，使其大小与最大的高度和宽度一致，并且根据需要返回像素掩码
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
        # 构建返回的数据字典，包含填充后的像素值
        data = {"pixel_values": padded_images}

        # 如果需要返回像素掩码
        if return_pixel_mask:
            # 对批次中的每张图像生成相应的像素掩码，并添加到数据字典中
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        # 返回一个 BatchFeature 对象，包含填充后的数据和指定类型的张量
        return BatchFeature(data=data, tensor_type=return_tensors)
```  
    # 预处理函数，用于对输入的图像数据进行预处理操作
    def preprocess(
        # 输入参数 images: 图像输入，可以是单张图像或图像列表
        images: ImageInput,
        # 是否进行调整大小的标志，可选参数，默认为 None
        do_resize: Optional[bool] = None,
        # 调整大小的目标尺寸，可选参数，默认为 None
        size: Optional[Dict[str, int]] = None,
        # 尺寸除数，可选参数，默认为 None
        size_divisor: Optional[int] = None,
        # 重采样方法，可选参数，默认为 None
        resample: PILImageResampling = None,
        # 是否进行重新缩放的标志，可选参数，默认为 None
        do_rescale: Optional[bool] = None,
        # 重新缩放因子，可选参数，默认为 None
        rescale_factor: Optional[float] = None,
        # 是否进行归一化的标志，可选参数，默认为 None
        do_normalize: Optional[bool] = None,
        # 图像均值，可选参数，默认为 None
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差，可选参数，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,
        # 是否进行填充的标志，可选参数，默认为 None
        do_pad: Optional[bool] = None,
        # 是否进行中心裁剪的标志，可选参数，默认为 None
        do_center_crop: Optional[bool] = None,
        # 返回张量的类型，可选参数，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式，通道维度的位置，可选参数，默认为 ChannelDimension.FIRST
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式，可选参数，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数
        **kwargs,
```