# `.\models\maskformer\feature_extraction_maskformer.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
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
"""
Feature extractor class for MaskFormer.
"""

import warnings  # 导入警告模块

from ...utils import logging  # 导入日志工具
from .image_processing_maskformer import MaskFormerImageProcessor  # 导入MaskFormerImageProcessor类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class MaskFormerFeatureExtractor(MaskFormerImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示MaskFormerFeatureExtractor类在将来的版本中将被移除
        warnings.warn(
            "The class MaskFormerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MaskFormerImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)  # 调用父类的构造函数，初始化MaskFormerImageProcessor的实例
```