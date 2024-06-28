# `.\models\deprecated\tapex\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING
# 导入延迟加载模块
from ....utils import _LazyModule

# 定义模块的导入结构，指定了 tokenization_tapex 模块中的 TapexTokenizer 类
_import_structure = {"tokenization_tapex": ["TapexTokenizer"]}

# 如果当前是类型检查阶段
if TYPE_CHECKING:
    # 导入 TapexTokenizer 类型，用于类型检查
    from .tokenization_tapex import TapexTokenizer

# 如果不是类型检查阶段（即运行时）
else:
    # 导入 sys 模块，用于后续模块替换操作
    import sys

    # 通过动态设置 sys.modules，将当前模块替换为延迟加载的 _LazyModule 对象
    # 这样可以延迟导入实际的模块内容，直到真正需要使用时才加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```