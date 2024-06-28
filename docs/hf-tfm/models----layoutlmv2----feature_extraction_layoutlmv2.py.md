# `.\models\layoutlmv2\feature_extraction_layoutlmv2.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Feature extractor class for LayoutLMv2.
"""

import warnings

from ...utils import logging
from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessor

# 获取全局日志记录器对象
logger = logging.get_logger(__name__)

# LayoutLMv2FeatureExtractor 类，继承自 LayoutLMv2ImageProcessor 类
class LayoutLMv2FeatureExtractor(LayoutLMv2ImageProcessor):
    
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 LayoutLMv2FeatureExtractor 类将在 Transformers 的第五个版本中弃用并移除，建议使用 LayoutLMv2ImageProcessor 类替代
        warnings.warn(
            "The class LayoutLMv2FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use LayoutLMv2ImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 LayoutLMv2ImageProcessor 的初始化方法，传入所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```