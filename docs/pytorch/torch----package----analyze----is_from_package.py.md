# `.\pytorch\torch\package\analyze\is_from_package.py`

```
# 导入模块类型和任意类型的类型提示
from types import ModuleType
from typing import Any

# 从当前模块的_mangling子模块导入is_mangled函数
from .._mangling import is_mangled

# 定义一个函数，用于判断对象是否来自于一个包(package)
def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    # 如果传入的对象是模块类型
    if type(obj) == ModuleType:
        # 则调用is_mangled函数检查模块名是否经过名称修饰（mangling）
        return is_mangled(obj.__name__)
    else:
        # 否则，使用传入对象的类型的模块名来调用is_mangled函数检查模块名是否经过名称修饰
        return is_mangled(type(obj).__module__)
```