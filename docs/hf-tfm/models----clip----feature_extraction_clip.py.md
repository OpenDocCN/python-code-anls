# `.\models\clip\feature_extraction_clip.py`

```py
# coding=utf-8
# 指定文件编码为UTF-8

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权

# you may not use this file except in compliance with the License.
# 除非符合许可证规定，否则不得使用此文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据“原样”分发软件，无论是明示的还是暗示的保证或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证以了解特定的语言授权和限制。

"""Feature extractor class for CLIP."""
# 用于 CLIP 的特征提取器类。

import warnings
# 导入警告模块

from ...utils import logging
# 导入日志记录工具模块

from .image_processing_clip import CLIPImageProcessor
# 导入 CLIP 图像处理模块中的 CLIPImageProcessor 类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

class CLIPFeatureExtractor(CLIPImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use CLIPImageProcessor instead.",
            FutureWarning,
        )
        # 发出警告，提醒 CLIPFeatureExtractor 类已废弃，将在 Transformers 版本 5 中移除，建议使用 CLIPImageProcessor 替代。

        super().__init__(*args, **kwargs)
        # 调用父类构造函数，初始化 CLIPImageProcessor 类
```