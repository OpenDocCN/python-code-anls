# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triinterpolate.pyi`

```
# 从 matplotlib.tri 模块中导入 Triangulation 和 TriFinder 类
from matplotlib.tri import Triangulation, TriFinder

# 导入 Literal 类型提示
from typing import Literal
import numpy as np
# 从 numpy.typing 模块导入 ArrayLike 类型提示
from numpy.typing import ArrayLike

# TriInterpolator 类定义，用于三角网格插值
class TriInterpolator:
    # 初始化方法
    def __init__(
        self,
        triangulation: Triangulation,  # 接受 Triangulation 对象作为三角网格参数
        z: ArrayLike,                  # 接受 z 数组作为数据参数
        trifinder: TriFinder | None = ...,  # 可选的 TriFinder 对象或者 None
    ) -> None: ...
    # __call__ 方法，实现了函数调用接口，返回 np.ma.MaskedArray 类型的结果
    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ma.MaskedArray: ...
    # gradient 方法，计算梯度，返回一个包含两个 np.ma.MaskedArray 类型的元组
    def gradient(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]: ...

# LinearTriInterpolator 类继承自 TriInterpolator 类，表示线性三角网格插值
class LinearTriInterpolator(TriInterpolator): ...

# CubicTriInterpolator 类继承自 TriInterpolator 类，表示立方体三角网格插值
class CubicTriInterpolator(TriInterpolator):
    # 初始化方法
    def __init__(
        self,
        triangulation: Triangulation,        # 接受 Triangulation 对象作为三角网格参数
        z: ArrayLike,                        # 接受 z 数组作为数据参数
        kind: Literal["min_E", "geom", "user"] = ...,  # 选择插值类型，可以是 "min_E"、"geom" 或 "user"
        trifinder: TriFinder | None = ...,   # 可选的 TriFinder 对象或者 None
        dz: tuple[ArrayLike, ArrayLike] | None = ...,  # 可选的 dz 元组参数，包含两个 ArrayLike 类型的数组或者 None
    ) -> None: ...
```