# `.\models\vipllava\__init__.py`

```
# 版权声明和许可信息，指明代码版权和许可协议
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# 导入可选依赖未可用的异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 模块导入结构定义
_import_structure = {"configuration_vipllava": ["VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP", "VipLlavaConfig"]}

# 检查是否可用 torch
try:
    if not is_torch_available():
        # 若不可用，则抛出可选依赖不可用异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 处理可选依赖不可用的情况
    pass
else:
    # 若可用 torch，则添加以下模块到导入结构中
    _import_structure["modeling_vipllava"] = [
        "VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VipLlavaForConditionalGeneration",
        "VipLlavaPreTrainedModel",
    ]

# 如果进行类型检查
if TYPE_CHECKING:
    # 从 configuration_vipllava 模块导入特定符号
    from .configuration_vipllava import VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP, VipLlavaConfig

    # 再次检查是否可用 torch
    try:
        if not is_torch_available():
            # 若不可用，则抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 处理可选依赖不可用的情况
        pass
    else:
        # 若可用 torch，则从 modeling_vipllava 模块导入特定符号
        from .modeling_vipllava import (
            VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            VipLlavaForConditionalGeneration,
            VipLlavaPreTrainedModel,
        )

# 如果不进行类型检查
else:
    # 导入 sys 模块
    import sys

    # 动态设置当前模块为 LazyModule，延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```