# `.\transformers\models\mluke\__init__.py`

```py
# 版权声明及许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 在 Apache 许可证 2.0 版本下进行许可
# 除非符合许可证的规定，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非必须按照适用法律或书面同意的方式，否则以“原样”分发的软件
# 没有任何种类的明示或暗示的担保或条件，包括但不限于
# 关于特定语言的特定用途的担保或条件
# 请查看许可证以获取关于特定语言的详细规定和限制

# 导入必要的模块和类
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义导入结构
_import_structure = {}

# 尝试导入 sentencepiece，如果不可用，则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mluke"] = ["MLukeTokenizer"]

# 如果需要类型检查
if TYPE_CHECKING:
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 tokenization_mluke 模块导入 MLukeTokenizer 类
        from .tokenization_mluke import MLukeTokenizer

# 如果不需要类型检查
else:
    import sys

    # 将当前模块设置为 LazyModule 类
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```