# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triangulation.pyi`

```
# 导入 _tri 模块中的 Triangulation 类
# 这个模块是 Matplotlib 的一个内部模块，用于处理三角剖分相关的功能
from matplotlib import _tri
# 从 matplotlib.tri._trifinder 模块中导入 TriFinder 类
# TriFinder 类用于在三角剖分中查找三角形和点之间的关系
from matplotlib.tri._trifinder import TriFinder

# 导入 numpy 库并将 ArrayLike 类型引入命名空间
import numpy as np
from numpy.typing import ArrayLike
# 导入 Any 类型以支持任意类型的数据
from typing import Any

# 定义 Triangulation 类
class Triangulation:
    # 定义 x、y 和 mask 属性，它们分别存储节点的 x、y 坐标和可选的遮罩
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray | None
    # 定义 is_delaunay 属性，表示是否是 Delaunay 三角剖分
    is_delaunay: bool
    # 定义 triangles 属性，存储三角形的索引数组
    triangles: np.ndarray

    # 构造函数，初始化 Triangulation 对象
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        triangles: ArrayLike | None = ...,
        mask: ArrayLike | None = ...,
    ) -> None: ...

    # 计算平面系数的方法，接受 z 数组作为输入，返回系数数组
    def calculate_plane_coefficients(self, z: ArrayLike) -> np.ndarray: ...

    # edges 属性，返回三角剖分的边界数组
    @property
    def edges(self) -> np.ndarray: ...

    # 获取 C++ 三角剖分对象的方法，返回 _tri.Triangulation 对象
    def get_cpp_triangulation(self) -> _tri.Triangulation: ...

    # 获取带有遮罩的三角形数组的方法
    def get_masked_triangles(self) -> np.ndarray: ...

    # 静态方法，从 args 和 kwargs 获取 Triangulation 对象以及其他参数和关键字参数的元组
    @staticmethod
    def get_from_args_and_kwargs(
        *args, **kwargs
    ) -> tuple[Triangulation, tuple[Any, ...], dict[str, Any]]: ...

    # 获取 TriFinder 对象的方法，用于在三角剖分中查找邻接关系
    def get_trifinder(self) -> TriFinder: ...

    # neighbors 属性，返回每个节点的邻居节点数组
    @property
    def neighbors(self) -> np.ndarray: ...

    # 设置遮罩的方法，接受 None 或者 ArrayLike 类型的 mask 参数
    def set_mask(self, mask: None | ArrayLike) -> None: ...
```