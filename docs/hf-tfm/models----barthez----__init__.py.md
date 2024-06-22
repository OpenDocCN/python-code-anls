# `.\transformers\models\barthez\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 导入自定义异常和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义模块的导入结构
_import_structure = {}

# 检查是否可用SentencePiece模块，若不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将BarthezTokenizer模块添加到导入结构中
    _import_structure["tokenization_barthez"] = ["BarthezTokenizer"]

# 检查是否可用Tokenizers模块，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将BarthezTokenizerFast模块添加到导入结构中
    _import_structure["tokenization_barthez_fast"] = ["BarthezTokenizerFast"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 检查是否可用SentencePiece模块，若不可用则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入BarthezTokenizer模块
        from .tokenization_barthez import BarthezTokenizer

    # 检查是否可用Tokenizers模块，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入BarthezTokenizerFast模块
        from .tokenization_barthez_fast import BarthezTokenizerFast

# 如果不是类型检查模式
else:
    # 导入系统模块
    import sys

    # 将当前模块重定向为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```