# `.\models\idefics\image_processing_idefics.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""Idefics 的图像处理器类。"""

from typing import Callable, Dict, List, Optional, Union

# 导入 PIL 库中的 Image 模块
from PIL import Image

# 导入相关的工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available

# IDEFICS 标准均值和标准差
IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]

# 将图像转换为 RGB 格式
def convert_to_rgb(image):
    # 如果图像已经是 RGB 格式，则直接返回
    if image.mode == "RGB":
        return image

    # 将图像转换为 RGBA 格式
    image_rgba = image.convert("RGBA")
    # 创建白色背景
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    # 处理透明图像的情况
    alpha_composite = Image.alpha_composite(background, image_rgba)
    # 将处理后的图像转换为 RGB 格式
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

# Idefics 图像处理器类，继承自 BaseImageProcessor
class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    构造一个 Idefics 图像处理器。

    Args:
        image_size (`int`, *optional*, defaults to 224):
            调整图像大小
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            如果对图像进行归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以在 `preprocess` 方法中的 `image_mean` 参数中覆盖。可以在 `preprocess` 方法中的 `image_mean` 参数中覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            如果对图像进行归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。
            可以在 `preprocess` 方法中的 `image_std` 参数中覆盖。可以在 `preprocess` 方法中的 `image_std` 参数中覆盖。
        image_num_channels (`int`, *optional*, defaults to 3):
            图像通道数。
    """

    # 模型输入名称列表
    model_input_names = ["pixel_values"]
    # 初始化函数，设置图像大小、均值、标准差、通道数等参数
    def __init__(
        self,
        image_size: int = 224,  # 图像大小，默认为224
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可选参数，可以是单个值或列表
        image_std: Optional[Union[float, List[float]] = None,  # 图像标准差，可选参数，可以是单个值或列表
        image_num_channels: Optional[int] = 3,  # 图像通道数，默认为3
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置图像大小、通道数、均值、标准差等参数
        self.image_size = image_size
        self.image_num_channels = image_num_channels
        self.image_mean = image_mean
        self.image_std = image_std

    # 图像预处理函数，对输入图像进行预处理
    def preprocess(
        self,
        images: ImageInput,  # 输入图像
        image_num_channels: Optional[int] = 3,  # 图像通道数，默认为3
        image_size: Optional[Dict[str, int]] = None,  # 图像大小的字典，可选参数
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可选参数，可以是单个值或列表
        image_std: Optional[Union[float, List[float]] = None,  # 图像标准差，可选参数，可以是单个值或列表
        transform: Callable = None,  # 可调用对象，用于对图像进行变换
        **kwargs,  # 其他关键字参数
```