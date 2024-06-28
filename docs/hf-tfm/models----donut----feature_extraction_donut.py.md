# `.\models\donut\feature_extraction_donut.py`

```
# 设置文件编码为UTF-8
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

"""Feature extractor class for Donut."""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 导入图像处理模块
from .image_processing_donut import DonutImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 DonutFeatureExtractor 类，继承自 DonutImageProcessor 类
class DonutFeatureExtractor(DonutImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提醒该类将在 Transformers 版本 5 中被移除，建议使用 DonutImageProcessor 代替
        warnings.warn(
            "The class DonutFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use DonutImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```