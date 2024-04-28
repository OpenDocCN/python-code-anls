# `.\transformers\models\phobert\__init__.py`

```
# 导入 TYPE_CHECKING 用于类型检查
from typing import TYPE_CHECKING
# 导入 _LazyModule 用于延迟加载模块
from ...utils import _LazyModule

# 定义模块的导入结构，包括 tokenization_phobert 模块下的 PhobertTokenizer 类
_import_structure = {"tokenization_phobert": ["PhobertTokenizer"]}

# 如果是类型检查，则执行以下语句块
if TYPE_CHECKING:
    # 从 tokenization_phobert 模块导入 PhobertTokenizer 类
    from .tokenization_phobert import PhobertTokenizer

# 如果不是类型检查，则执行以下语句块
else:
    # 导入 sys 模块
    import sys
    # 将当前模块替换为 LazyModule，实现延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```