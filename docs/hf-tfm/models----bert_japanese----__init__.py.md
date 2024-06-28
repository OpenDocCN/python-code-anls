# `.\models\bert_japanese\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入 LazyModule 用于延迟加载模块
from ...utils import _LazyModule

# 定义要导入的结构，包括 tokenization_bert_japanese 模块的几个特定类
_import_structure = {"tokenization_bert_japanese": ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"]}

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入具体的类，以便类型检查器能够正确处理类型
    from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer

# 如果不是在进行类型检查
else:
    # 导入 sys 模块以便后续使用
    import sys

    # 将当前模块替换为 LazyModule，以便在需要时才加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```