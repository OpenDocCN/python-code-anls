# `.\models\layoutlmv3\feature_extraction_layoutlmv3.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for LayoutLMv3.
"""

import warnings  # 导入警告模块

from ...utils import logging  # 导入日志工具
from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor  # 导入图像处理的LayoutLMv3类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class LayoutLMv3FeatureExtractor(LayoutLMv3ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class LayoutLMv3FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use LayoutLMv3ImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)  # 调用父类构造函数，初始化LayoutLMv3ImageProcessor的实例
```