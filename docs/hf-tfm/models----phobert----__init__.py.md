# `.\models\phobert\__init__.py`

```py
# 版权声明和许可证信息，指明此代码的版权归 HuggingFace Team 所有，依据 Apache License, Version 2.0 发布
#
# 在符合许可证的情况下，可以使用此文件。您可以在以下链接获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件根据“原样”分发，不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

# 导入类型检查模块中的 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从 utils 模块中导入 LazyModule 类
from ...utils import _LazyModule

# 定义模块的导入结构
_import_structure = {"tokenization_phobert": ["PhobertTokenizer"]}

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从 tokenization_phobert 模块中导入 PhobertTokenizer 类型
    from .tokenization_phobert import PhobertTokenizer

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块指定为 LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```