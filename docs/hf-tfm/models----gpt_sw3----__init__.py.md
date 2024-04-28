# `.\models\gpt_sw3\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入自定义的异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义模块的导入结构
_import_structure = {}

# 尝试导入 sentencepiece 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果可用，则将 GPTSw3Tokenizer 添加到导入结构中
else:
    _import_structure["tokenization_gpt_sw3"] = ["GPTSw3Tokenizer"]

# 如果是类型检查的情况
if TYPE_CHECKING:
    try:
        # 再次尝试导入 sentencepiece 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 GPTSw3Tokenizer
        from .tokenization_gpt_sw3 import GPTSw3Tokenizer
# 如果不是类型检查的情况
else:
    # 导入 sys 模块
    import sys
    # 将当前模块的字典形式添加到全局变量的 __name__ 模块中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```