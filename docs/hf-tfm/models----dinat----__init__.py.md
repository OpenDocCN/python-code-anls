# `.\models\dinat\__init__.py`

```
# 版权声明和许可证声明，指明版权所有者及许可协议
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

# 导入类型检查模块中的 TYPE_CHECKING
from typing import TYPE_CHECKING

# 导入必要的依赖项
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {"configuration_dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"]}

# 检查是否有必要的 Torch 依赖，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果依赖可用，扩展导入结构以包含 DINAT 模型相关的内容
    _import_structure["modeling_dinat"] = [
        "DINAT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DinatForImageClassification",
        "DinatModel",
        "DinatPreTrainedModel",
        "DinatBackbone",
    ]

# 如果是类型检查阶段，则导入 DINAT 配置和模型相关的内容
if TYPE_CHECKING:
    from .configuration_dinat import DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP, DinatConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dinat import (
            DINAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DinatBackbone,
            DinatForImageClassification,
            DinatModel,
            DinatPreTrainedModel,
        )

# 如果不是类型检查阶段，则将当前模块设为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```