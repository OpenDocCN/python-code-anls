# `.\transformers\models\mbart50\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available

# 定义模块导入结构
_import_structure = {}

# 检查是否安装了 sentencepiece 库
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mbart50"] = ["MBart50Tokenizer"]

# 检查是否安装了 tokenizers 库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mbart50_fast"] = ["MBart50TokenizerFast"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 检查是否安装了 sentencepiece 库
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 MBart50Tokenizer 类
        from .tokenization_mbart50 import MBart50Tokenizer

    # 检查是否安装了 tokenizers 库
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 MBart50TokenizerFast 类
        from .tokenization_mbart50_fast import MBart50TokenizerFast

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```