# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_trifinder.pyi`

```
from matplotlib.tri import Triangulation
from numpy.typing import ArrayLike

class TriFinder:
    # TriFinder类，用于查找点在三角网格中的三角形索引
    def __init__(self, triangulation: Triangulation) -> None:
        # TriFinder类的初始化方法，接收一个Triangulation对象作为参数
        ...

    def __call__(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        # TriFinder类的调用方法，接收两个ArrayLike类型的参数x和y，返回一个ArrayLike类型的结果
        ...


class TrapezoidMapTriFinder(TriFinder):
    # TrapezoidMapTriFinder类，继承自TriFinder类，用于在梯形映射中查找点在三角网格中的三角形索引
    def __init__(self, triangulation: Triangulation) -> None:
        # TrapezoidMapTriFinder类的初始化方法，接收一个Triangulation对象作为参数
        ...

    def __call__(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        # TrapezoidMapTriFinder类的调用方法，接收两个ArrayLike类型的参数x和y，返回一个ArrayLike类型的结果
        ...
```