# `.\models\deit\feature_extraction_deit.py`

```py
# 导入 utf-8 编码
# coding=utf-8
# 导入 HuggingFace Inc. 团队的版权声明
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# 导入 Apache License, Version 2.0 的许可声明
# Licensed under the Apache License, Version 2.0 (the "License");
# 说明只有在遵守 License 的情况下才能使用该代码
# you may not use this file except in compliance with the License.
# 说明如何获取 License
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 说明如果不遵守 License，代码将以"原样"分发，没有任何形式的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 说明 License 中的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 描述该文件的功能：DeiT 的特征提取器类
"""Feature extractor class for DeiT."""

# 导入警告模块
import warnings

# 导入日志记录器
from ...utils import logging
# 导入 DeiTImageProcessor 类
from .image_processing_deit import DeiTImageProcessor


# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义 DeiTFeatureExtractor 类，继承自 DeiTImageProcessor 类
class DeiTFeatureExtractor(DeiTImageProcessor):
    # 初始化函数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，说明该类已被弃用，将在 Transformers 的第 5 版中删除
        warnings.warn(
            "The class DeiTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use DeiTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)
```