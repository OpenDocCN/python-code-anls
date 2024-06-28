# `.\models\mobilenet_v2\feature_extraction_mobilenet_v2.py`

```py
# coding=utf-8
# 定义了编码格式为 UTF-8，确保支持多种字符集
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，声明了代码的版权信息和保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 指定本代码采用 Apache License, Version 2.0 授权许可
# you may not use this file except in compliance with the License.
# 除非符合许可证的要求，否则不得使用本文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证以查看特定语言下的权限和限制

"""Feature extractor class for MobileNetV2."""
# 为 MobileNetV2 设计的特征提取器类

import warnings
# 导入警告模块，用于发出警告消息

from ...utils import logging
# 导入日志工具模块中的 logging 函数
from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor
# 从当前目录下的 image_processing_mobilenet_v2 模块中导入 MobileNetV2ImageProcessor 类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

class MobileNetV2FeatureExtractor(MobileNetV2ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class MobileNetV2FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileNetV2ImageProcessor instead.",
            FutureWarning,
        )
        # 发出警告，提醒用户 MobileNetV2FeatureExtractor 类已经废弃，并将在 Transformers 版本 5 中移除
        # 建议用户使用 MobileNetV2ImageProcessor 替代

        super().__init__(*args, **kwargs)
        # 调用父类 MobileNetV2ImageProcessor 的初始化方法，传递所有参数
```