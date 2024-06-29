# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\__init__.pyi`

```
# 从 typing 模块中导入 TypeVar 类型变量
from typing import TypeVar

# 从 _axes 模块中导入 Axes 类，并将其命名为 Subplot，以保持向后兼容性
from ._axes import Axes as Subplot

# 定义一个类型变量 _T，用于泛型类型注解
_T = TypeVar("_T")

# Backcompat.
# 将 Subplot 别名指向 Axes，以保持向后兼容性
Subplot = Axes

# 定义一个元类 _SubplotBaseMeta，用于创建 subplot 类的基础元类
class _SubplotBaseMeta(type):
    # 定义 __instancecheck__ 方法，用于检查对象是否属于该类型
    def __instancecheck__(self, obj) -> bool: ...

# 定义 SubplotBase 类，使用 _SubplotBaseMeta 作为其元类
class SubplotBase(metaclass=_SubplotBaseMeta): ...

# 定义 subplot 类工厂函数 subplot_class_factory，接受一个类型参数 cls，返回相同类型的类
def subplot_class_factory(cls: type[_T]) -> type[_T]: ...
```