# `.\transformers\models\byt5\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入惰性模块
from ...utils import _LazyModule

# 定义需要导入的结构
_import_structure = {"tokenization_byt5": ["ByT5Tokenizer"]}

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从子模块中导入 ByT5Tokenizer 类型
    from .tokenization_byt5 import ByT5Tokenizer
# 如果不是类型检查模式
else:
    # 引入 sys 模块
    import sys
    # 将当前模块替换为惰性模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```