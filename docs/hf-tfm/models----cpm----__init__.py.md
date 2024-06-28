# `.\models\cpm\__init__.py`

```py
# 引入类型检查模块，用于判断当前是否处于类型检查模式
from typing import TYPE_CHECKING

# 引入自定义的异常类，用于处理可选依赖项不可用的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义一个空的导入结构字典，用于存储延迟导入的模块和类
_import_structure = {}

# 尝试检查是否可用 sentencepiece，如果不可用则抛出自定义异常 OptionalDependencyNotAvailable
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 CpmTokenizer 添加到导入结构中
    _import_structure["tokenization_cpm"] = ["CpmTokenizer"]

# 尝试检查是否可用 tokenizers，如果不可用则抛出自定义异常 OptionalDependencyNotAvailable
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 CpmTokenizerFast 添加到导入结构中
    _import_structure["tokenization_cpm_fast"] = ["CpmTokenizerFast"]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    try:
        # 再次检查是否可用 sentencepiece，如果不可用则抛出自定义异常 OptionalDependencyNotAvailable
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，在类型检查模式下从 tokenization_cpm 导入 CpmTokenizer
        from .tokenization_cpm import CpmTokenizer

    try:
        # 再次检查是否可用 tokenizers，如果不可用则抛出自定义异常 OptionalDependencyNotAvailable
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，在类型检查模式下从 tokenization_cpm_fast 导入 CpmTokenizerFast
        from .tokenization_cpm_fast import CpmTokenizerFast

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设为一个延迟加载模块，使用 _LazyModule 将 _import_structure 作为导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```