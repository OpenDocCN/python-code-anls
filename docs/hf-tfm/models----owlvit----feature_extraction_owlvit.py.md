# `.\transformers\models\owlvit\feature_extraction_owlvit.py`

```
# coding=utf-8
# 声明使用 utf-8 编码

# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 声明版权所有者和版权保护范围

# Licensed under the Apache License, Version 2.0 (the "License");
# 声明基于 Apache License 2.0 进行许可

# you may not use this file except in compliance with the License.
# 表示必须遵守许可证的规定才能使用该代码

# You may obtain a copy of the License at
# 声明可以从以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证的网址

# Unless required by applicable law or agreed to in writing, software
# 表示如果没有相关法律要求，软件

# distributed under the License is distributed on an "AS IS" BASIS,
# 在许可证下分发时是"按原样"分发的

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# 查看许可证中对权限和

# limitations under the License.
# 限制的具体规定

# 为 OwlViT 定义特征提取器类的模块

import warnings

# 导入 warnings 模块用于发出警告

from ...utils import logging
# 从 utils 模块导入 logging 功能

from .image_processing_owlvit import OwlViTImageProcessor
# 从 image_processing_owlvit 模块导入 OwlViTImageProcessor 类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

class OwlViTFeatureExtractor(OwlViTImageProcessor):
    # 定义 OwlViTFeatureExtractor 类，继承自 OwlViTImageProcessor 类
    def __init__(self, *args, **kwargs) -> None:
        # 初始化方法
        warnings.warn(
            # 发出警告
            "The class OwlViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use OwlViTImageProcessor instead.",
            # 警告信息
            FutureWarning,
            # 警告类型
        )
        super().__init__(*args, **kwargs)
        # 调用父类 OwlViTImageProcessor 的初始化方法
```