# `.\models\layoutxlm\__init__.py`

```
# 引入必要的类型检查模块
from typing import TYPE_CHECKING

# 从工具包中引入所需的依赖项和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构
_import_structure = {"processing_layoutxlm": ["LayoutXLMProcessor"]}

# 检查是否安装了句子分词工具，如果没有则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 LayoutXLMTokenizer 导入结构中
    _import_structure["tokenization_layoutxlm"] = ["LayoutXLMTokenizer"]

# 检查是否安装了 Tokenizers 库，如果没有则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 LayoutXLMTokenizerFast 导入结构中
    _import_structure["tokenization_layoutxlm_fast"] = ["LayoutXLMTokenizerFast"]

# 如果正在进行类型检查，执行以下操作
if TYPE_CHECKING:
    # 从当前模块中导入 LayoutXLMProcessor 类
    from .processing_layoutxlm import LayoutXLMProcessor

    # 再次检查句子分词工具是否可用，如果可用则从相应模块导入 LayoutXLMTokenizer 类
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutxlm import LayoutXLMTokenizer

    # 再次检查 Tokenizers 是否可用，如果可用则从相应模块导入 LayoutXLMTokenizerFast 类
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutxlm_fast import LayoutXLMTokenizerFast

# 如果不是类型检查阶段，则进行模块的懒加载处理
else:
    # 导入 sys 模块
    import sys

    # 创建一个 LazyModule 对象，将当前模块注册为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```