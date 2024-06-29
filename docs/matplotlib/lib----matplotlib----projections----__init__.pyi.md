# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\__init__.pyi`

```py
from .geo import (
    AitoffAxes as AitoffAxes,  # 导入 AitoffAxes 类并重命名为 AitoffAxes
    HammerAxes as HammerAxes,  # 导入 HammerAxes 类并重命名为 HammerAxes
    LambertAxes as LambertAxes,  # 导入 LambertAxes 类并重命名为 LambertAxes
    MollweideAxes as MollweideAxes,  # 导入 MollweideAxes 类并重命名为 MollweideAxes
)
from .polar import PolarAxes as PolarAxes  # 导入 PolarAxes 类并重命名为 PolarAxes
from ..axes import Axes  # 导入 Axes 类

class ProjectionRegistry:
    def __init__(self) -> None: ...  # 空的构造函数，无需额外注释
    def register(self, *projections: type[Axes]) -> None: ...  # 注册投影类型的方法，参数为 Axes 类型的可变参数
    def get_projection_class(self, name: str) -> type[Axes]: ...  # 根据名称获取投影类的方法，返回 Axes 类型
    def get_projection_names(self) -> list[str]: ...  # 获取所有投影名称的方法，返回字符串列表

projection_registry: ProjectionRegistry  # 声明一个 ProjectionRegistry 类的实例变量

def register_projection(cls: type[Axes]) -> None: ...  # 注册投影类的函数，参数为 Axes 类型
def get_projection_class(projection: str | None = ...) -> type[Axes]: ...  # 获取投影类的函数，参数可选字符串，返回 Axes 类型
def get_projection_names() -> list[str]: ...  # 获取所有投影名称的函数，返回字符串列表
```