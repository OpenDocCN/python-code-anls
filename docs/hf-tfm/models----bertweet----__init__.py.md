# `.\models\bertweet\__init__.py`

```py
# 版权声明和许可证信息
# 版权所有 2020 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 软件没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入延迟加载模块
from ...utils import _LazyModule

# 定义模块的导入结构
_import_structure = {"tokenization_bertweet": ["BertweetTokenizer"]}

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从本地模块中导入 BertweetTokenizer 类型
    from .tokenization_bertweet import BertweetTokenizer

# 如果不是类型检查阶段（即运行阶段）
else:
    # 导入系统模块
    import sys

    # 将当前模块指定为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```