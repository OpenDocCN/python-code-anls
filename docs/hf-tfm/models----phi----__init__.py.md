# `.\transformers\models\phi\__init__.py`

```
# 版权声明及许可协议信息
# Copyright 2023 Microsoft and The HuggingFace Inc. team. All rights reserved.
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


# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入实用工具函数和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_phi": ["PHI_PRETRAINED_CONFIG_ARCHIVE_MAP", "PhiConfig"],
}

# 检查是否有 Torch 库可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，添加模型相关模块到导入结构中
    _import_structure["modeling_phi"] = [
        "PHI_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PhiPreTrainedModel",
        "PhiModel",
        "PhiForCausalLM",
        "PhiForSequenceClassification",
        "PhiForTokenClassification",
    ]


# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关模块
    from .configuration_phi import PHI_PRETRAINED_CONFIG_ARCHIVE_MAP, PhiConfig

    # 再次检查 Torch 库是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，导入模型相关模块
        from .modeling_phi import (
            PHI_PRETRAINED_MODEL_ARCHIVE_LIST,
            PhiForCausalLM,
            PhiForSequenceClassification,
            PhiForTokenClassification,
            PhiModel,
            PhiPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule 对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```