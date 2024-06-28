# `.\models\kosmos2\__init__.py`

```
# coding=utf-8
# 文件编码声明，使用 UTF-8 编码格式
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 授权许可，采用 Apache 许可证 2.0 版本
# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 依据许可证分发的软件是基于"原样"分发的，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# 请参阅许可证以了解特定语言的权限
# limitations under the License.
# 许可证下的限制

from typing import TYPE_CHECKING
# 引入类型检查模块

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)
# 从相对路径中引入依赖模块和函数

_import_structure = {
    "configuration_kosmos2": ["KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Kosmos2Config"],
    "processing_kosmos2": ["Kosmos2Processor"],
}
# 定义导入结构字典，包含模块名称和对应的导入内容列表

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_kosmos2"] = [
        "KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Kosmos2ForConditionalGeneration",
        "Kosmos2Model",
        "Kosmos2PreTrainedModel",
    ]
# 尝试导入 torch 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常并忽略，否则将模型相关内容加入导入结构字典

if TYPE_CHECKING:
    from .configuration_kosmos2 import KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP, Kosmos2Config
    from .processing_kosmos2 import Kosmos2Processor
    # 如果在类型检查模式下，则从相对路径导入配置和处理器模块内容

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_kosmos2 import (
            KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Kosmos2ForConditionalGeneration,
            Kosmos2Model,
            Kosmos2PreTrainedModel,
        )
        # 如果在类型检查模式下且 torch 可用，则从相对路径导入模型相关内容

else:
    import sys
    # 如果不在类型检查模式下，则导入系统模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
    # 将当前模块注册为懒加载模块，指定模块名、文件名和导入结构
```