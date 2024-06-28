# `.\models\stablelm\__init__.py`

```py
# 版权声明及许可信息
# Copyright 2024 Stability AI and The HuggingFace Inc. team. All rights reserved.
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

# 导入 TYPE_CHECKING 模块
from typing import TYPE_CHECKING

# 导入 OptionalDependencyNotAvailable 异常和 _LazyModule 类，以及 is_torch_available 函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_stablelm": ["STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP", "StableLmConfig"],
}

# 检查是否导入了 torch 库，如果未导入，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 torch 库，则添加以下模块到导入结构中
    _import_structure["modeling_stablelm"] = [
        "StableLmForCausalLM",
        "StableLmModel",
        "StableLmPreTrainedModel",
        "StableLmForSequenceClassification",
    ]

# 如果当前环境是类型检查环境
if TYPE_CHECKING:
    # 导入 configuration_stablelm 模块中的特定内容
    from .configuration_stablelm import STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP, StableLmConfig

    # 再次检查是否导入了 torch 库，如果未导入，则继续抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果导入了 torch 库，则导入 modeling_stablelm 模块中的特定内容
        from .modeling_stablelm import (
            StableLmForCausalLM,
            StableLmForSequenceClassification,
            StableLmModel,
            StableLmPreTrainedModel,
        )

# 如果当前不是类型检查环境
else:
    import sys

    # 将当前模块替换为 _LazyModule 实例，支持懒加载模块的功能
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```