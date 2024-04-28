# `.\transformers\models\nougat\__init__.py`

```
# 从 typing 模块导入 TYPE_CHECKING 常量
from typing import TYPE_CHECKING
# 从 utils 模块导入 OptionalDependencyNotAvailable 类和 _LazyModule 函数，以及 is_tokenizers_available 和 is_vision_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_vision_available

# 定义一个字典结构，包含了要导入的模块及其成员
_import_structure = {
    "processing_nougat": ["NougatProcessor"],  # 导入 processing_nougat 模块，并包含其中的 NougatProcessor 类
}

# 尝试检查是否 Tokenizers 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Tokenizers 库可用，则将 NougatTokenizerFast 类添加到导入结构中
    _import_structure["tokenization_nougat_fast"] = ["NougatTokenizerFast"]

# 尝试检查是否 Vision 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Vision 库可用，则将 NougatImageProcessor 类添加到导入结构中
    _import_structure["image_processing_nougat"] = ["NougatImageProcessor"]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 processing_nougat 模块导入 NougatProcessor 类
    from .processing_nougat import NougatProcessor

    # 尝试检查是否 Tokenizers 库可用，若不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Tokenizers 库可用，则从 tokenization_nougat_fast 模块导入 NougatTokenizerFast 类
        from .tokenization_nougat_fast import NougatTokenizerFast

    # 尝试检查是否 Vision 库可用，若不可用则忽略
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Vision 库可用，则从 image_processing_nougat 模块导入 NougatImageProcessor 类
        from .image_processing_nougat import NougatImageProcessor

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块的名称、文件路径和导入结构传递给 _LazyModule 函数，将模块替换为惰性模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```