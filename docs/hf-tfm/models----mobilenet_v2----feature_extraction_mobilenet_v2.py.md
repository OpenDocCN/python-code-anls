# `.\transformers\models\mobilenet_v2\feature_extraction_mobilenet_v2.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明：2022年HuggingFace Inc.团队。保留所有权利。
#
# 根据 Apache License, Version 2.0 许可协议，禁止未经许可使用此文件
# 可以在遵守许可协议的情况下使用本文件，可以获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"原样"分发的，
# 没有任何担保或条件，无论明示或暗示。
# 有关特定语言规定权限和限制，请查看许可协议
"""MobileNetV2的特征提取器类。"""

# 导入警告模块
import warnings

# 从工具函数中导入日志记录模块
from ...utils import logging
# 从图像处理模块中导入MobileNetV2图像处理类
from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor

# 获取本模块的日志记录器
logger = logging.get_logger(__name__)

# MobileNetV2特征提取器类继承自MobileNetV2图像处理类
class MobileNetV2FeatureExtractor(MobileNetV2ImageProcessor):
    # 初始化方法，接受任意数量的参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告消息，标记MobileNetV2FeatureExtractor类已被弃用，并将在Transformers的第5个版本中移除
        warnings.warn(
            "The class MobileNetV2FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileNetV2ImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类MobileNetV2ImageProcessor的初始化方法
        super().__init__(*args, **kwargs)
```