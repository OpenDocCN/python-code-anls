# `.\models\herbert\__init__.py`

```
# 版权声明及许可声明，说明代码受 Apache 许可证 2.0 版本保护
#
# 从 typing 模块导入 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从 ...utils 中导入必要的异常和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available

# 定义模块导入结构
_import_structure = {"tokenization_herbert": ["HerbertTokenizer"]}

# 检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加快速 tokenization_herbert_fast 的导入结构
    _import_structure["tokenization_herbert_fast"] = ["HerbertTokenizerFast"]


# 如果在类型检查环境下
if TYPE_CHECKING:
    # 从 .tokenization_herbert 模块导入 HerbertTokenizer 类
    from .tokenization_herbert import HerbertTokenizer

    # 再次检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 .tokenization_herbert_fast 模块导入 HerbertTokenizerFast 类
        from .tokenization_herbert_fast import HerbertTokenizerFast

# 如果不在类型检查环境下
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule，使用 LazyModule 对象初始化模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```