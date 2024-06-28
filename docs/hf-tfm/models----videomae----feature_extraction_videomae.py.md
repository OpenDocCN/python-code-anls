# `.\models\videomae\feature_extraction_videomae.py`

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
"""
Feature extractor class for VideoMAE.
"""

# 导入警告模块
import warnings
# 导入日志模块
from ...utils import logging
# 导入图像处理类 VideoMAEImageProcessor
from .image_processing_videomae import VideoMAEImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# VideoMAEFeatureExtractor 类继承自 VideoMAEImageProcessor 类
class VideoMAEFeatureExtractor(VideoMAEImageProcessor):
    
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出关于类即将在 Transformers 版本 5 中移除的警告
        warnings.warn(
            "The class VideoMAEFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use VideoMAEImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 VideoMAEImageProcessor 的初始化方法
        super().__init__(*args, **kwargs)
```