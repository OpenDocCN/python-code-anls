# `.\models\code_llama\__init__.py`

```
# 版权声明以及许可信息
# Copyright 2023 MetaAI and The HuggingFace Inc. team. All rights reserved.
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

# 引入类型检查
from typing import TYPE_CHECKING

# 引入导入所需的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

_import_structure = {}

# 检查是否有sentencepiece可用
try:
    if not is_sentencepiece_available():
        # 如果不可用，则引发OptionalDependencyNotAvailable错误
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_code_llama"] = ["CodeLlamaTokenizer"]

# 检查是否有tokenizers可用
try:
    if not is_tokenizers_available():
        # 如果不可用，则引发OptionalDependencyNotAvailable错误
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_code_llama_fast"] = ["CodeLlamaTokenizerFast"]

# 如果处于类型检查模式
if TYPE_CHECKING:
    try:
        if not is_sentencepiece_available():
            # 如果不可用，则引发OptionalDependencyNotAvailable错误
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入CodeLlamaTokenizer
        from .tokenization_code_llama import CodeLlamaTokenizer

    try:
        if not is_tokenizers_available():
            # 如果不可用，则引发OptionalDependencyNotAvailable错误
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入CodeLlamaTokenizerFast
        from .tokenization_code_llama_fast import CodeLlamaTokenizerFast

# 如果不处于类型检查模式
else:
    import sys

    # 将该模块添加到sys.modules，这样就可以按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```