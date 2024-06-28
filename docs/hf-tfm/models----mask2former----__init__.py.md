# `.\models\mask2former\__init__.py`

```
# 引入模块的版权声明和许可证信息
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入必要的依赖项检查函数和模块
# 从当前目录的utils模块中引入相关函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构，包含配置和模型
_import_structure = {
    "configuration_mask2former": [
        "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Mask2FormerConfig",
    ],
}

# 检查是否存在视觉处理依赖，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将视觉处理相关模块添加到导入结构中
    _import_structure["image_processing_mask2former"] = ["Mask2FormerImageProcessor"]

# 检查是否存在PyTorch依赖，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将模型处理相关模块添加到导入结构中
    _import_structure["modeling_mask2former"] = [
        "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Mask2FormerForUniversalSegmentation",
        "Mask2FormerModel",
        "Mask2FormerPreTrainedModel",
    ]

# 如果当前环境支持类型检查，引入配置相关的类和变量
if TYPE_CHECKING:
    from .configuration_mask2former import MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, Mask2FormerConfig

    # 检查视觉处理是否可用，如果可用则引入相关处理类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_mask2former import Mask2FormerImageProcessor

    # 检查PyTorch是否可用，如果可用则引入相关模型类和变量
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mask2former import (
            MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )

# 如果当前环境不支持类型检查，将模块设置为LazyModule以支持按需导入
else:
    import sys

    # 将当前模块替换为LazyModule，用于按需加载导入的模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```