# `.\models\cohere\__init__.py`

```
# 版权声明和许可证信息，指明代码版权所有者和使用许可
# Copyright 2024 Cohere and The HuggingFace Inc. team. All rights reserved.
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

# 导入需要的类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖项和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_cohere": ["COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP", "CohereConfig"],
}

# 检查是否有 tokenizers 库可用，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_cohere_fast 模块到导入结构中
    _import_structure["tokenization_cohere_fast"] = ["CohereTokenizerFast"]

# 检查是否有 torch 库可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_cohere 模块到导入结构中
    _import_structure["modeling_cohere"] = [
        "CohereForCausalLM",
        "CohereModel",
        "CoherePreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入配置和模型相关的模块
    from .configuration_cohere import COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP, CohereConfig

    # 检查是否有 tokenizers 库可用，若不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 tokenization_cohere_fast 模块
        from .tokenization_cohere_fast import CohereTokenizerFast

    # 检查是否有 torch 库可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 modeling_cohere 模块中的类
        from .modeling_cohere import (
            CohereForCausalLM,
            CohereModel,
            CoherePreTrainedModel,
        )

# 如果不是类型检查环境
else:
    # 导入 sys 模块
    import sys

    # 使用 _LazyModule 将当前模块作为惰性模块导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```