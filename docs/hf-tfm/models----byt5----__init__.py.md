# `.\models\byt5\__init__.py`

```py
# 导入必要的模块和类型检查工具
from typing import TYPE_CHECKING
# 导入懒加载模块
from ...utils import _LazyModule

# 定义模块导入结构，指定要导入的模块和类
_import_structure = {"tokenization_byt5": ["ByT5Tokenizer"]}

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 ByT5Tokenizer 类型
    from .tokenization_byt5 import ByT5Tokenizer
# 如果不在类型检查模式下
else:
    # 导入 sys 模块
    import sys
    # 将当前模块替换为懒加载模块，使用 LazyModule 将当前模块名、文件名、导入结构、模块规范传入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```