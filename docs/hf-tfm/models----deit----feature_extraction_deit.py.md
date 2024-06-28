# `.\models\deit\feature_extraction_deit.py`

```py
# coding=utf-8
# 声明编码格式为 UTF-8

# 版权声明
# 版权所有 © 2021 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证。

"""DeiT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging

# 导入 DeiT 图像处理模块
from .image_processing_deit import DeiTImageProcessor

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# DeiTFeatureExtractor 类继承自 DeiTImageProcessor 类
class DeiTFeatureExtractor(DeiTImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 DeiTFeatureExtractor 类已被弃用，并将在 Transformers 版本 5 中移除
        warnings.warn(
            "The class DeiTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use DeiTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```