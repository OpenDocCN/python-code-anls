# `.\models\poolformer\image_processing_poolformer.py`

```
# 设置编码格式为UTF-8
# 版权声明和许可证信息
# 版权归The HuggingFace Inc.团队所有，保留所有权利。
# 根据Apache License 2.0许可证使用本文件，除非符合许可证中的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件，
# 没有任何明示或暗示的担保或条件。详细信息请参阅许可证。
"""PoolFormer的图像处理类。"""

from typing import Dict, List, Optional, Union

import numpy as np

# 导入图像处理相关的工具和库
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
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging

# 如果视觉库可用，则导入PIL库
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)


# PoolFormer图像处理器类，继承自BaseImageProcessor
class PoolFormerImageProcessor(BaseImageProcessor):
    r"""
    构造一个PoolFormer图像处理器。

    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]

    # 初始化方法，设置图像处理器的各种参数
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
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # 如果传入的尺寸参数为None，则设定默认值为{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 调用函数get_size_dict，获取调整尺寸的字典，允许非正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果传入的裁剪尺寸参数为None，则设定默认值为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 调用函数get_size_dict，获取裁剪尺寸的字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置是否进行调整尺寸的标志
        self.do_resize = do_resize
        # 设置调整尺寸的参数字典
        self.size = size
        # 设置裁剪比例
        self.crop_pct = crop_pct
        # 设置重采样方法
        self.resample = resample
        # 设置是否进行中心裁剪的标志
        self.do_center_crop = do_center_crop
        # 设置裁剪尺寸的参数字典
        self.crop_size = crop_size
        # 设置是否进行重新缩放的标志
        self.do_rescale = do_rescale
        # 设置重新缩放因子
        self.rescale_factor = rescale_factor
        # 设置是否进行归一化的标志
        self.do_normalize = do_normalize
        # 设置图像均值，如果未指定则使用默认值IMAGENET_DEFAULT_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 设置图像标准差，如果未指定则使用默认值IMAGENET_DEFAULT_STD
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 设置有效的处理器关键字列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "crop_pct",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
```