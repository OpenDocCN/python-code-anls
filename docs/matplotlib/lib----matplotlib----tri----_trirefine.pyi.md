# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_trirefine.pyi`

```py
from typing import Literal, overload  # 导入类型提示工具

import numpy as np  # 导入NumPy库
from numpy.typing import ArrayLike  # 导入NumPy的ArrayLike类型

from matplotlib.tri._triangulation import Triangulation  # 导入Triangulation类
from matplotlib.tri._triinterpolate import TriInterpolator  # 导入TriInterpolator类

class TriRefiner:
    def __init__(self, triangulation: Triangulation) -> None: ...
    # 初始化TriRefiner类，接受一个Triangulation对象作为参数

class UniformTriRefiner(TriRefiner):
    def __init__(self, triangulation: Triangulation) -> None: ...
    # 初始化UniformTriRefiner类，继承自TriRefiner类，接受一个Triangulation对象作为参数

    @overload
    def refine_triangulation(
        self, *, return_tri_index: Literal[True], subdiv: int = ...
    ) -> tuple[Triangulation, np.ndarray]: ...
    # 方法重载：对三角化进行细化，返回三角化对象和三角化索引数组

    @overload
    def refine_triangulation(
        self, return_tri_index: Literal[False] = ..., subdiv: int = ...
    ) -> Triangulation: ...
    # 方法重载：对三角化进行细化，返回三角化对象

    @overload
    def refine_triangulation(
        self, return_tri_index: bool = ..., subdiv: int = ...
    ) -> tuple[Triangulation, np.ndarray] | Triangulation: ...
    # 方法重载：对三角化进行细化，根据return_tri_index的值返回不同类型的结果

    def refine_field(
        self,
        z: ArrayLike,
        triinterpolator: TriInterpolator | None = ...,
        subdiv: int = ...,
    ) -> tuple[Triangulation, np.ndarray]: ...
    # 对字段进行细化，返回三角化对象和数据数组
```