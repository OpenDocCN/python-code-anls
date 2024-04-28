# `.\transformers\models\wav2vec2_phoneme\__init__.py`

```
# 版权声明和许可声明
# 版权声明和许可声明

# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入延迟模块
from ...utils import _LazyModule

# 定义导入结构
_import_structure = {"tokenization_wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"]}

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从tokenization_wav2vec2_phoneme模块中导入Wav2Vec2PhonemeCTCTokenizer类
    from .tokenization_wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
# 如果不是类型检查模式
else:
    # 导入sys模块
    import sys
    # 将当前模块注册为延迟模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```