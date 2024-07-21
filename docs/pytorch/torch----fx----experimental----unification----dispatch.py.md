# `.\pytorch\torch\fx\experimental\unification\dispatch.py`

```py
# 导入 functools 模块中的 partial 函数，用于创建偏函数
from functools import partial
# 从当前模块的 multipledispatch 中导入 dispatch 函数，并忽略类型检查的警告
from .multipledispatch import dispatch  # type: ignore[import]

# 定义一个空的字典作为命名空间
namespace = {}  # type: ignore[var-annotated]

# 使用 functools.partial 函数创建一个新的 partial 函数 dispatch，并将命名空间参数设置为上面定义的空字典
dispatch = partial(dispatch, namespace=namespace)
```