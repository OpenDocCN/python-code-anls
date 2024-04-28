# `.\models\layoutxlm\__init__.py`

```
# 从 typing 模块中导入 TYPE_CHECKING 常量，用于检查类型
from typing import TYPE_CHECKING

# 从当前包的 utils 模块中导入相关函数和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构，用于延迟加载
_import_structure = {"processing_layoutxlm": ["LayoutXLMProcessor"]}

# 检查是否 SentencePiece 库可用
try:
    if not is_sentencepiece_available():
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_layoutxlm 到导入结构中
    _import_structure["tokenization_layoutxlm"] = ["LayoutXLMTokenizer"]

# 检查是否 Tokenizers 库可用
try:
    if not is_tokenizers_available():
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_layoutxlm_fast 到导入结构中
    _import_structure["tokenization_layoutxlm_fast"] = ["LayoutXLMTokenizerFast"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从当前包中的 processing_layoutxlm 模块导入 LayoutXLMProcessor 类
    from .processing_layoutxlm import LayoutXLMProcessor

    # 再次检查是否 SentencePiece 库可用
    try:
        if not is_sentencepiece_available():
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从当前包中的 tokenization_layoutxlm 模块导入 LayoutXLMTokenizer 类
        from .tokenization_layoutxlm import LayoutXLMTokenizer

    # 再次检查是否 Tokenizers 库可用
    try:
        if not is_tokenizers_available():
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从当前包中的 tokenization_layoutxlm_fast 模块导入 LayoutXLMTokenizerFast 类
        from .tokenization_layoutxlm_fast import LayoutXLMTokenizerFast

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 使用 _LazyModule 类将当前模块设为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```