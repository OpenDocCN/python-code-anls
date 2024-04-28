# `.\transformers\models\mobilenet_v1\feature_extraction_mobilenet_v1.py`

```py
# 设置编码格式为utf-8
# 版权声明：2022年HuggingFace Inc.团队。保留所有权利。
# 根据Apache许可证2.0版（“许可证”）获授权。您只能在遵守许可证的情况下使用此文件。
# 您可以获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则软件
# 根据许可证在“原样”的基础上分发，没有任何明示或暗示的保证或条件。
# 有关特定语言的权限和限制限制，请参阅许可证。
"""MobileNetV1的特征提取器类。"""

import warnings
# 导入日志记录工具
from ...utils import logging
# 从image_processing_mobilenet_v1模块导入MobileNetV1ImageProcessor类

from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义MobileNetV1FeatureExtractor类，继承自MobileNetV1ImageProcessor类
class MobileNetV1FeatureExtractor(MobileNetV1ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示MobileNetV1FeatureExtractor类已被弃用并将在Transformers的第5版本中移除
        # 请使用MobileNetV1ImageProcessor类
        warnings.warn(
            "The class MobileNetV1FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileNetV1ImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类MobileNetV1ImageProcessor的构造方法
        super().__init__(*args, **kwargs)
```