# `.\models\herbert\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available

# 定义模块的导入结构
_import_structure = {"tokenization_herbert": ["HerbertTokenizer"]}

# 检查是否存在 tokenizers 库，如果不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 库，则添加快速版本的 HerbertTokenizer 到导入结构中
    _import_structure["tokenization_herbert_fast"] = ["HerbertTokenizerFast"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入 HerbertTokenizer 类
    from .tokenization_herbert import HerbertTokenizer

    # 再次检查是否存在 tokenizers 库，如果不存在则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 tokenizers 库，则导入 HerbertTokenizerFast 类
        from .tokenization_herbert_fast import HerbertTokenizerFast

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```