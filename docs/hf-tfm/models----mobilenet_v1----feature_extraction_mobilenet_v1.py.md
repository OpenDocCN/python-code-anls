# `.\models\mobilenet_v1\feature_extraction_mobilenet_v1.py`

```py
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
"""Feature extractor class for MobileNetV1."""

import warnings  # 导入警告模块

from ...utils import logging  # 导入日志工具
from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor  # 导入MobileNetV1的图像处理类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class MobileNetV1FeatureExtractor(MobileNetV1ImageProcessor):  # 定义MobileNetV1特征提取器类，继承自MobileNetV1ImageProcessor类
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class MobileNetV1FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileNetV1ImageProcessor instead.",
            FutureWarning,  # 发出FutureWarning警告，提示MobileNetV1FeatureExtractor类将在Transformers的第五个版本中被移除，建议使用MobileNetV1ImageProcessor类
        )
        super().__init__(*args, **kwargs)  # 调用父类MobileNetV1ImageProcessor的初始化方法
```