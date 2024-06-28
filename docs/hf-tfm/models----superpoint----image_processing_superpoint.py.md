# `.\models\superpoint\image_processing_superpoint.py`

```
# 版权声明和许可证信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""SuperPoint 的图像处理类。"""

from typing import Dict, Optional, Union  # 导入类型提示模块

import numpy as np  # 导入NumPy库

from ... import is_vision_available, requires_backends  # 导入可视化模块相关函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理相关函数和类
from ...image_transforms import resize, to_channel_dimension_format  # 导入图像变换函数
from ...image_utils import (  # 导入图像工具函数
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging  # 导入工具函数和日志模块

if is_vision_available():  # 检查视觉模块是否可用
    import PIL  # 如果可用，则导入PIL库

logger = logging.get_logger(__name__)  # 获取日志记录器

def is_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    # 检查图像是否为灰度图像的函数
    if input_data_format == ChannelDimension.FIRST:  # 如果输入数据格式是通道维度在最前面
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])  # 检查RGB通道是否相同
    elif input_data_format == ChannelDimension.LAST:  # 如果输入数据格式是通道维度在最后面
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])  # 检查RGB通道是否相同

def convert_to_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    """
    使用NTSC公式将图像转换为灰度格式。仅支持numpy和PIL Image。TODO：支持torch和tensorflow的灰度转换

    该函数本应返回一个单通道图像，但由于在https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446中讨论的问题，
    它返回一个每个通道都具有相同值的三通道图像。

    Args:
        image (Image):
            要转换的图像。
        input_data_format (`ChannelDimension`或`str`，*可选*):
            输入图像的通道维度格式。
    """
    requires_backends(convert_to_grayscale, ["vision"])  # 要求后端支持视觉处理

    if isinstance(image, np.ndarray):  # 如果图像是NumPy数组
        if input_data_format == ChannelDimension.FIRST:  # 如果输入数据格式是通道维度在最前面
            # 使用NTSC公式将RGB图像转换为灰度图像
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=0)  # 将灰度图像复制为三通道的形式
        elif input_data_format == ChannelDimension.LAST:  # 如果输入数据格式是通道维度在最后面
            # 使用NTSC公式将RGB图像转换为灰度图像
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=-1)  # 将灰度图像复制为三通道的形式
        return gray_image  # 返回灰度图像
    # 如果参数 `image` 不是 `PIL.Image.Image` 类型的实例，则直接返回该参数
    if not isinstance(image, PIL.Image.Image):
        return image
    
    # 将图像转换为灰度图像
    image = image.convert("L")
    
    # 返回转换后的图像
    return image
# 定义一个名为 SuperPointImageProcessor 的类，继承自 BaseImageProcessor 类
class SuperPointImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SuperPoint image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            控制是否将图像的高度和宽度调整为指定的 `size`。可以在 `preprocess` 方法中被 `do_resize` 覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 480, "width": 640}`):
            调整后的输出图像的分辨率。仅在 `do_resize` 设置为 `True` 时有效。可以在 `preprocess` 方法中被 `size` 覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的比例因子 `rescale_factor` 对图像进行重新缩放。可以在 `preprocess` 方法中被 `do_rescale` 覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，则使用的比例因子。可以在 `preprocess` 方法中被 `rescale_factor` 覆盖。
    """

    # 定义类属性 model_input_names，表示模型输入的名称列表，这里只包含 "pixel_values"
    model_input_names = ["pixel_values"]

    # 定义初始化方法，接受一系列参数来配置 SuperPointImageProcessor 的实例
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        **kwargs,
    ) -> None:
        # 调用父类 BaseImageProcessor 的初始化方法，并传递额外的关键字参数
        super().__init__(**kwargs)
        
        # 如果未提供 size 参数，则使用默认的 {"height": 480, "width": 640}
        size = size if size is not None else {"height": 480, "width": 640}
        
        # 调用辅助函数 get_size_dict，根据默认设置将 size 转换为包含高度和宽度的字典，确保不强制为正方形
        size = get_size_dict(size, default_to_square=False)

        # 将参数赋值给实例变量
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

    # 定义 resize 方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"height": int, "width": int}`, specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be inferred from the input
                image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        # 根据指定的大小字典，获取调整后的大小（可能会保持长宽比）
        size = get_size_dict(size, default_to_square=False)

        # 调用 resize 函数进行图像调整，传入调整后的大小、数据格式和其他关键字参数
        return resize(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```