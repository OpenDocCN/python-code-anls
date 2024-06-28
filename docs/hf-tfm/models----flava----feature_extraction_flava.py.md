# `.\models\flava\feature_extraction_flava.py`

```
# coding=utf-8
# 文件编码声明为 UTF-8

# 版权声明及许可证信息
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
"""FLAVA 的特征提取器类。"""

# 引入警告模块
import warnings

# 引入日志模块
from ...utils import logging
# 引入 FLAVA 图像处理模块中的特征提取器类
from .image_processing_flava import FlavaImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# FLAVA 特征提取器类，继承自 FlavaImageProcessor 类
class FlavaFeatureExtractor(FlavaImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，表明 FlavaFeatureExtractor 类已弃用，将在 Transformers 版本 5 中移除，建议使用 FlavaImageProcessor 替代
        warnings.warn(
            "The class FlavaFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use FlavaImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```