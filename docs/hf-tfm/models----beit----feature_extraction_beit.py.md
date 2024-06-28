# `.\models\beit\feature_extraction_beit.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 2021年HuggingFace团队。版权所有。
#
# 根据Apache许可证2.0版（“许可证”）许可；您不得使用此文件，除非符合许可证的规定。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于“原样”提供，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
"""BEiT的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 导入BEiT图像处理器类
from .image_processing_beit import BeitImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义BEiT特征提取器类，继承自BeitImageProcessor类
class BeitFeatureExtractor(BeitImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示BeitFeatureExtractor类即将在Transformers版本5中删除，请使用BeitImageProcessor代替
        warnings.warn(
            "The class BeitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use BeitImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```