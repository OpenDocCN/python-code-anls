# `.\models\tvp\__init__.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本许可

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件均为“按原样”分发，不附带任何担保或条件，无论是明示的还是默示的

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细了解许可证以了解权限和限制

from typing import TYPE_CHECKING

# 从特定位置导入依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_tvp": [
        "TVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TvpConfig",
    ],
    "processing_tvp": ["TvpProcessor"],
}

# 尝试导入视觉处理相关模块，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_tvp"] = ["TvpImageProcessor"]

# 尝试导入 Torch 相关模块，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tvp"] = [
        "TVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TvpModel",
        "TvpPreTrainedModel",
        "TvpForVideoGrounding",
    ]

# 如果是类型检查模式，导入特定的模块
if TYPE_CHECKING:
    from .configuration_tvp import (
        TVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvpConfig,
    )
    from .processing_tvp import TvpProcessor

    # 尝试导入视觉处理相关模块，如果不可用则跳过
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_tvp import TvpImageProcessor

    # 尝试导入 Torch 相关模块，如果不可用则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tvp import (
            TVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TvpForVideoGrounding,
            TvpModel,
            TvpPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块设置为延迟加载模块
else:
    import sys

    # 设置当前模块为 LazyModule，根据特定的导入结构和规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```