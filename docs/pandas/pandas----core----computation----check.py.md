# `D:\src\scipysrc\pandas\pandas\core\computation\check.py`

```
# 导入用于动态导入依赖的函数 import_optional_dependency
from pandas.compat._optional import import_optional_dependency

# 尝试导入 numexpr 库，如果失败则发出警告并返回 None
ne = import_optional_dependency("numexpr", errors="warn")

# 检查是否成功导入 numexpr 库，将结果保存在 NUMEXPR_INSTALLED 变量中
NUMEXPR_INSTALLED = ne is not None

# 定义 __all__ 列表，包含当前模块的公开接口名称 NUMEXPR_INSTALLED
__all__ = ["NUMEXPR_INSTALLED"]
```