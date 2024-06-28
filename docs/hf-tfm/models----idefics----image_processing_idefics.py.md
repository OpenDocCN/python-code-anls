# `.\models\idefics\image_processing_idefics.py`

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
"""Image processor class for Idefics."""

from typing import Callable, Dict, List, Optional, Union

from PIL import Image  # 导入 PIL 库中的 Image 模块

from ...image_processing_utils import BaseImageProcessor, BatchFeature  # 导入自定义的图像处理工具
from ...image_transforms import resize, to_channel_dimension_format  # 导入自定义的图像转换函数
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)  # 导入图像处理和转换的实用函数
from ...utils import TensorType, is_torch_available  # 导入通用实用函数和 Torch 相关函数

IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]  # 定义 IDEFICS 标准均值
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]  # 定义 IDEFICS 标准标准差


def convert_to_rgb(image):
    # `image.convert("RGB")` 只对 .jpg 图片有效，因为它会为透明图像创建错误的背景。
    # `alpha_composite` 函数处理带有透明通道的图像。
    if image.mode == "RGB":  # 检查图像是否已经是 RGB 模式
        return image

    image_rgba = image.convert("RGBA")  # 将图像转换为 RGBA 模式
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))  # 创建白色背景图像
    alpha_composite = Image.alpha_composite(background, image_rgba)  # 使用 alpha 合成处理透明通道
    alpha_composite = alpha_composite.convert("RGB")  # 将结果转换回 RGB 模式
    return alpha_composite


class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        image_num_channels (`int`, *optional*, defaults to 3):
            Number of image channels.
    """

    model_input_names = ["pixel_values"]  # 模型输入的名称列表，此处只有一个像素值的输入
    # 初始化方法，用于设置图像处理的参数和调用父类的初始化方法
    def __init__(
        self,
        image_size: int = 224,                            # 图像大小，默认为224像素
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可以是单个数值或列表形式的均值
        image_std: Optional[Union[float, List[float]]] = None,   # 图像标准差，可以是单个数值或列表形式的标准差
        image_num_channels: Optional[int] = 3,             # 图像通道数，默认为3通道（彩色图像）
        **kwargs,                                          # 其他关键字参数
    ) -> None:
        # 调用父类的初始化方法，处理其他传入的关键字参数
        super().__init__(**kwargs)

        # 设置对象的属性值，用于后续图像预处理使用
        self.image_size = image_size                       # 设置图像大小
        self.image_num_channels = image_num_channels       # 设置图像通道数
        self.image_mean = image_mean                       # 设置图像均值
        self.image_std = image_std                         # 设置图像标准差

    # 图像预处理方法，用于对输入图像进行预处理操作
    def preprocess(
        self,
        images: ImageInput,                                # 输入的图像数据，可以是单张图像或批量图像
        image_num_channels: Optional[int] = 3,             # 图像通道数，默认为3通道
        image_size: Optional[Dict[str, int]] = None,       # 图像大小的字典，包含宽和高
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可以是单个数值或列表形式的均值
        image_std: Optional[Union[float, List[float]]] = None,   # 图像标准差，可以是单个数值或列表形式的标准差
        transform: Callable = None,                        # 图像变换函数，用于额外的图像处理
        **kwargs,                                          # 其他关键字参数
```