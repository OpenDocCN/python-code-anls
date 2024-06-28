# `.\models\barthez\__init__.py`

```py
# 导入类型检查模块中的 TYPE_CHECKING
from typing import TYPE_CHECKING

# 导入自定义的异常类 OptionalDependencyNotAvailable 和延迟加载模块 _LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule

# 导入用于检查依赖是否可用的函数 is_sentencepiece_available 和 is_tokenizers_available
from ...utils import is_sentencepiece_available, is_tokenizers_available

# 定义一个空的字典 _import_structure 用于存储导入结构
_import_structure = {}

# 检查是否 sentencepiece 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 "tokenization_barthez" 映射到 ["BarthezTokenizer"] 并存入 _import_structure
    _import_structure["tokenization_barthez"] = ["BarthezTokenizer"]

# 检查是否 tokenizers 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 "tokenization_barthez_fast" 映射到 ["BarthezTokenizerFast"] 并存入 _import_structure
    _import_structure["tokenization_barthez_fast"] = ["BarthezTokenizerFast"]

# 如果 TYPE_CHECKING 为 True，则执行以下导入语句
if TYPE_CHECKING:
    # 检查是否 sentencepiece 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 tokenization_barthez 模块导入 BarthezTokenizer 类
        from .tokenization_barthez import BarthezTokenizer

    # 检查是否 tokenizers 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 tokenization_barthez_fast 模块导入 BarthezTokenizerFast 类
        from .tokenization_barthez_fast import BarthezTokenizerFast

# 如果 TYPE_CHECKING 为 False（通常为运行时），则执行以下导入语句
else:
    import sys

    # 动态地将当前模块指定为延迟加载模块 _LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```