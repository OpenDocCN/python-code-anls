# `.\transformers\models\bert_japanese\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入延迟加载模块
from ...utils import _LazyModule

# 模块的导入结构，包含了需要导入的内容及其组织结构
_import_structure = {"tokenization_bert_japanese": ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"]}

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从.tokenization_bert_japanese模块中导入指定的类
    from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer

# 如果不是类型检查阶段
else:
    # 导入sys模块
    import sys
    # 将当前模块替换为一个延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```