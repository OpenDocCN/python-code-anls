# `.\models\vit\feature_extraction_vit.py`

```py
# coding=utf-8
# 定义文件编码格式为 UTF-8

# 版权声明和许可证信息，指定代码的版权和使用条款
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature extractor class for ViT."""
# 导入警告模块，用于向用户显示警告信息
import warnings

# 导入 logging 模块，用于记录日志
from ...utils import logging
# 从本地模块中导入 ViTImageProcessor 类
from .image_processing_vit import ViTImageProcessor

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义 ViTFeatureExtractor 类，继承自 ViTImageProcessor 类
class ViTFeatureExtractor(ViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出关于类被弃用的警告信息，建议使用 ViTImageProcessor 类代替
        warnings.warn(
            "The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ViTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的构造函数，传入所有参数和关键字参数
        super().__init__(*args, **kwargs)
```