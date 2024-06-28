# `.\models\nllb\__init__.py`

```
# 版权声明和许可信息
#
# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 在法律许可的范围内提供，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个空的导入结构
_import_structure = {}

# 检查是否可用 SentencePiece
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 NllbTokenizer 添加到导入结构中
    _import_structure["tokenization_nllb"] = ["NllbTokenizer"]

# 检查是否可用 Tokenizers
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 NllbTokenizerFast 添加到导入结构中
    _import_structure["tokenization_nllb_fast"] = ["NllbTokenizerFast"]

# 如果在类型检查模式下
if TYPE_CHECKING:
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 NllbTokenizer 类型
        from .tokenization_nllb import NllbTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 NllbTokenizerFast 类型
        from .tokenization_nllb_fast import NllbTokenizerFast

# 如果不在类型检查模式下
else:
    import sys

    # 将当前模块设为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```