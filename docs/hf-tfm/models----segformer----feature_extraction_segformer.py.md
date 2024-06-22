# `.\transformers\models\segformer\feature_extraction_segformer.py`

```py
# coding=utf-8
# 设置该文件的编码格式为 UTF-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 声明版权归 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 版本的许可证进行授权
# you may not use this file except in compliance with the License.
# 在遵守许可证的情况下才能使用该代码
# You may obtain a copy of the License at
# 您可以在以下位置获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证分发，是"按原样"分发的
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 查看许可证中规定的特定于许可证的权限和
# limitations under the License.
# 限制。
"""Feature extractor class for SegFormer."""
# 用于 SegFormer 的特征提取器类

import warnings
# 导入警告模块

from ...utils import logging
# 从工具包中导入日志记录模块
from .image_processing_segformer import SegformerImageProcessor
# 从 image_processing_segformer 模块导入 SegformerImageProcessor 类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

class SegformerFeatureExtractor(SegformerImageProcessor):
    # 定义 SegformerFeatureExtractor 类，继承自 SegformerImageProcessor 类
    def __init__(self, *args, **kwargs) -> None:
        # 定义构造函数，接受任意数量的位置参数和关键字参数
        warnings.warn(
            # 发出一个警告
            "The class SegformerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            # 警告信息
            " Please use SegformerImageProcessor instead.",
            # 警告建议
            FutureWarning,
        )
        # 设置警告类型为 FutureWarning
        super().__init__(*args, **kwargs)
        # 调用父类 SegformerImageProcessor 的构造函数
```