# `.\transformers\models\bertweet\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入惰性加载模块
from ...utils import _LazyModule

# 定义模块的导入结构，包括要导入的模块和对象
_import_structure = {"tokenization_bertweet": ["BertweetTokenizer"]}

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入 BertweetTokenizer 类型
    from .tokenization_bertweet import BertweetTokenizer

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 使用惰性加载模块，将当前模块替换为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```