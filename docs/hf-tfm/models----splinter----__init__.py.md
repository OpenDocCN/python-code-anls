# `.\models\splinter\__init__.py`

```py
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

# 导入类型检查工具
from typing import TYPE_CHECKING

# 导入所需的辅助函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_splinter": ["SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SplinterConfig"],
    "tokenization_splinter": ["SplinterTokenizer"],
}

# 检查是否可用 tokenizers 库，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 tokenization_splinter_fast 到导入结构，用于快速 tokenization
    _import_structure["tokenization_splinter_fast"] = ["SplinterTokenizerFast"]

# 检查是否可用 torch 库，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 modeling_splinter 到导入结构，包含多个 Splinter 模型组件
    _import_structure["modeling_splinter"] = [
        "SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SplinterForQuestionAnswering",
        "SplinterForPreTraining",
        "SplinterLayer",
        "SplinterModel",
        "SplinterPreTrainedModel",
    ]

# 如果是类型检查模式，则进行详细导入
if TYPE_CHECKING:
    from .configuration_splinter import SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP, SplinterConfig
    from .tokenization_splinter import SplinterTokenizer

    # 检查是否可用 tokenizers 库，若可用则进一步导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_splinter_fast import SplinterTokenizerFast

    # 检查是否可用 torch 库，若可用则进一步导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_splinter import (
            SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SplinterForPreTraining,
            SplinterForQuestionAnswering,
            SplinterLayer,
            SplinterModel,
            SplinterPreTrainedModel,
        )

# 非类型检查模式下，设置当前模块为惰性加载模块
else:
    import sys

    # 使用 _LazyModule 类来实现惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```