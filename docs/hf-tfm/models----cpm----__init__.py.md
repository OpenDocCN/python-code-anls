# `.\models\cpm\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入自定义的异常类OptionalDependencyNotAvailable和懒加载模块_LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义空的导入结构
_import_structure = {}

# 尝试检查是否可用sentencepiece，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果可用则将CpmTokenizer添加到导入结构
else:
    _import_structure["tokenization_cpm"] = ["CpmTokenizer"]

# 尝试检查是否可用tokenizers，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果可用则将CpmTokenizerFast添加到导入结构
else:
    _import_structure["tokenization_cpm_fast"] = ["CpmTokenizerFast"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 尝试检查是否可用sentencepiece，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果可用则从tokenization_cpm导入CpmTokenizer
    else:
        from .tokenization_cpm import CpmTokenizer

    # 尝试检查是否可用tokenizers，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果可用则从tokenization_cpm_fast导入CpmTokenizerFast
    else:
        from .tokenization_cpm_fast import CpmTokenizerFast
# 如果不是类型检查模式
else:
    # 导入sys模块
    import sys
    # 将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```