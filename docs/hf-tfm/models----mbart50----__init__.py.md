# `.\models\mbart50\__init__.py`

```
# 版权声明和许可信息，指明代码的版权和使用许可
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

# 导入必要的依赖项：OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义一个空的导入结构字典
_import_structure = {}

# 尝试导入 MBart50Tokenizer，如果 sentencepiece 不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，则将 MBart50Tokenizer 加入到 _import_structure 中
    _import_structure["tokenization_mbart50"] = ["MBart50Tokenizer"]

# 尝试导入 MBart50TokenizerFast，如果 tokenizers 不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，则将 MBart50TokenizerFast 加入到 _import_structure 中
    _import_structure["tokenization_mbart50_fast"] = ["MBart50TokenizerFast"]

# 如果是在类型检查环境中
if TYPE_CHECKING:
    try:
        # 再次尝试导入 MBart50Tokenizer，如果 sentencepiece 不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果成功导入，则从 tokenization_mbart50 中导入 MBart50Tokenizer
        from .tokenization_mbart50 import MBart50Tokenizer

    try:
        # 再次尝试导入 MBart50TokenizerFast，如果 tokenizers 不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果成功导入，则从 tokenization_mbart50_fast 中导入 MBart50TokenizerFast
        from .tokenization_mbart50_fast import MBart50TokenizerFast

# 如果不在类型检查环境中
else:
    # 导入 sys 模块
    import sys

    # 将当前模块指定为 _LazyModule 的实例，延迟加载模块的导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```