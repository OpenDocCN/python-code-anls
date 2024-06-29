# `.\numpy\numpy\compat\__init__.py`

```py
"""
python
"""
兼容性模块。

此模块包含从 Python 本身或第三方扩展复制的重复代码，可能包含以下原因：

  * 兼容性
  * 我们可能只需要复制库/模块的一小部分

此模块自 1.26.0 版本起已被弃用，并将在将来的版本中移除。

"""

# 导入警告模块
import warnings
# 从内部工具模块导入 _inspect 模块
from .._utils import _inspect
# 从内部工具模块的 _inspect 模块导入 getargspec 和 formatargspec 函数
from .._utils._inspect import getargspec, formatargspec
# 从 . 模块导入 py3k 模块
from . import py3k
# 从 .py3k 模块导入所有内容
from .py3k import *

# 引发警告，指示 np.compat 在 Python 2 到 3 的转换期间使用，自 1.26.0 版本起已弃用，并将被移除
warnings.warn(
    "`np.compat`, which was used during the Python 2 to 3 transition,"
    " is deprecated since 1.26.0, and will be removed",
    DeprecationWarning, stacklevel=2
)

# 将空列表tion,"
    " is deprecated since 1.26.0, and will be removed",
    DeprecationWarning, stacklevel=2
)

# 初始化模块的公开接口列表
__all__ = []

# 将 _inspect 模块中定义的所有公开名称添加到 __all__ 中
__all__.extend(_inspect.__all__)

# 将 py3k 模块中定义的所有公开名称添加到 __all__ 中
__all__.extend(py3k.__all__)
```