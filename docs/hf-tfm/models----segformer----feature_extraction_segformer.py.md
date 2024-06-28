# `.\models\segformer\feature_extraction_segformer.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8，确保可以处理中文等特殊字符
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 版权声明，声明版权归 HuggingFace Inc. 团队所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 Version 2.0 进行许可，即只要符合条件，可以自由使用、复制、修改、分发该代码
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不能使用此文件
# You may obtain a copy of the License at
# 您可以获取许可证的副本的地址
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证分发的软件是在“按原样”基础上分发的，即没有任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的担保或条件，明示的或隐含的
# See the License for the specific language governing permissions and
# 请查看许可证以了解特定语言下的权限
# limitations under the License.
# 限制和许可证下的责任

"""Feature extractor class for SegFormer."""
# 用于 SegFormer 的特征提取器类

import warnings
# 引入警告模块

from ...utils import logging
# 从当前包的父目录进入 utils 模块中导入 logging 模块
from .image_processing_segformer import SegformerImageProcessor
# 从当前目录下的 image_processing_segformer 模块中导入 SegformerImageProcessor 类


logger = logging.get_logger(__name__)
# 获取当前模块的 logger 对象，用于记录日志

class SegformerFeatureExtractor(SegformerImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 初始化方法
        warnings.warn(
            "The class SegformerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use SegformerImageProcessor instead.",
            FutureWarning,
        )
        # 发出警告，提示 SegformerFeatureExtractor 类已经弃用，并将在 Transformers 版本 5 中移除，建议使用 SegformerImageProcessor 替代
        super().__init__(*args, **kwargs)
        # 调用父类 SegformerImageProcessor 的初始化方法，传入所有位置参数和关键字参数
```