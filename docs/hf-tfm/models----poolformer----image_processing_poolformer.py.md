# `.\transformers\models\poolformer\image_processing_poolformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""PoolFormer 的图像处理器类。"""

from typing import Dict, List, Optional, Union

import numpy as np

# 导入相关的图像处理工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_vision_available, logging

# 如果视觉库可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# PoolFormer 图像处理器类，继承自 BaseImageProcessor 类
class PoolFormerImageProcessor(BaseImageProcessor):
    r"""
    构建一个 PoolFormer 图像处理器。

    """

    # 模型输入的名称
    model_input_names = ["pixel_values"]

    # 初始化方法
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        crop_pct: int = 0.9,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 设置默认的图像大小和裁剪大小
        size = size if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 初始化各种参数
        self.do_resize = do_resize
        self.size = size
        self.crop_pct = crop_pct
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
    # 定义一个方法用于调整图像大小
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据，类型为 numpy 数组
        size: Dict[str, int],  # 目标大小，字典类型，包含宽度和高度
        crop_pct: Optional[float] = None,  # 可选参数，裁剪比例，默认为 None
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，可选参数，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选参数，默认为 None
        **kwargs,  # 其他关键字参数
    # 定义一个方法用于预处理图像
    def preprocess(
        self,
        images: ImageInput,  # 输入的图像数据，可以是单个图像或图像列表
        do_resize: bool = None,  # 是否调整大小，默认为 None
        size: Dict[str, int] = None,  # 目标大小，字典类型，包含宽度和高度，默认为 None
        crop_pct: int = None,  # 裁剪比例，默认为 None
        resample: PILImageResampling = None,  # 重采样方法，默认为 None
        do_center_crop: bool = None,  # 是否中心裁剪，默认为 None
        crop_size: Dict[str, int] = None,  # 裁剪大小，字典类型，包含宽度和高度，默认为 None
        do_rescale: bool = None,  # 是否重新缩放，默认为 None
        rescale_factor: float = None,  # 重新缩放因子，默认为 None
        do_normalize: bool = None,  # 是否归一化，默认为 None
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可选参数，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，可选参数，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型，可选参数，默认为 None
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 数据格式，通道维度在前，默认为 FIRST
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选参数，默认为 None
        **kwargs,  # 其他关键字参数
```