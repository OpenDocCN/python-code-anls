# `.\models\code_llama\__init__.py`

```
# 导入所需的模块和函数
from typing import TYPE_CHECKING
# 导入自定义异常类，用于处理缺少可选依赖的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义模块的导入结构
_import_structure = {}

# 检查是否安装了 SentencePiece 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 CodeLlamaTokenizer 添加到模块的导入结构中
    _import_structure["tokenization_code_llama"] = ["CodeLlamaTokenizer"]

# 检查是否安装了 Tokenizers 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 CodeLlamaTokenizerFast 添加到模块的导入结构中
    _import_structure["tokenization_code_llama_fast"] = ["CodeLlamaTokenizerFast"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 tokenization_code_llama 模块中导入 CodeLlamaTokenizer 类
        from .tokenization_code_llama import CodeLlamaTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 tokenization_code_llama_fast 模块中导入 CodeLlamaTokenizerFast 类
        from .tokenization_code_llama_fast import CodeLlamaTokenizerFast

# 如果不是类型检查阶段，则为模块创建 LazyModule，并将其加入到 sys.modules 中
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```