# `D:\src\scipysrc\matplotlib\lib\matplotlib\bezier.pyi`

```py
from collections.abc import Callable  # 导入 Callable 类，用于函数签名类型提示
from typing import Literal  # 导入 Literal 类型，用于定义特定值的类型

import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.typing import ArrayLike  # 导入 ArrayLike 类型，用于表示类似数组的类型

from .path import Path  # 从当前包的 path 模块导入 Path 类

class NonIntersectingPathException(ValueError): ...  # 定义非交叉路径异常类，继承自 ValueError

def get_intersection(
    cx1: float,
    cy1: float,
    cos_t1: float,
    sin_t1: float,
    cx2: float,
    cy2: float,
    cos_t2: float,
    sin_t2: float,
) -> tuple[float, float]: ...  # 定义获取两条路径交点的函数签名

def get_normal_points(
    cx: float, cy: float, cos_t: float, sin_t: float, length: float
) -> tuple[float, float, float, float]: ...  # 定义获取法线点的函数签名

def split_de_casteljau(beta: ArrayLike, t: float) -> tuple[np.ndarray, np.ndarray]: ...  # 定义分割 De Casteljau 算法的函数签名

def find_bezier_t_intersecting_with_closedpath(
    bezier_point_at_t: Callable[[float], tuple[float, float]],
    inside_closedpath: Callable[[tuple[float, float]], bool],
    t0: float = ...,
    t1: float = ...,
    tolerance: float = ...,
) -> tuple[float, float]: ...  # 定义查找与闭合路径相交的贝塞尔曲线参数 t 的函数签名

# TODO make generic over d, the dimension? ndarraydim
class BezierSegment:
    def __init__(self, control_points: ArrayLike) -> None: ...  # 初始化贝塞尔片段对象，传入控制点数组
    def __call__(self, t: ArrayLike) -> np.ndarray: ...  # 定义贝塞尔片段对象的调用方法，返回点的数组
    def point_at_t(self, t: float) -> tuple[float, ...]: ...  # 获取给定参数 t 处的点的函数签名
    @property
    def control_points(self) -> np.ndarray: ...  # 返回贝塞尔片段对象的控制点数组的属性方法签名
    @property
    def dimension(self) -> int: ...  # 返回贝塞尔片段对象的维度属性方法签名
    @property
    def degree(self) -> int: ...  # 返回贝塞尔片段对象的阶数属性方法签名
    @property
    def polynomial_coefficients(self) -> np.ndarray: ...  # 返回贝塞尔片段对象的多项式系数数组的属性方法签名
    def axis_aligned_extrema(self) -> tuple[np.ndarray, np.ndarray]: ...  # 返回贝塞尔片段对象的轴对齐极值的函数签名

def split_bezier_intersecting_with_closedpath(
    bezier: ArrayLike,
    inside_closedpath: Callable[[tuple[float, float]], bool],
    tolerance: float = ...,
) -> tuple[np.ndarray, np.ndarray]: ...  # 定义分割与闭合路径相交的贝塞尔曲线的函数签名

def split_path_inout(
    path: Path,
    inside: Callable[[tuple[float, float]], bool],
    tolerance: float = ...,
    reorder_inout: bool = ...,
) -> tuple[Path, Path]: ...  # 定义分割路径为内部与外部部分的函数签名

def inside_circle(
    cx: float, cy: float, r: float
) -> Callable[[tuple[float, float]], bool]: ...  # 定义检查点是否在圆内的函数签名

def get_cos_sin(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]: ...  # 获取给定两点的余弦和正弦值的函数签名

def check_if_parallel(
    dx1: float, dy1: float, dx2: float, dy2: float, tolerance: float = ...
) -> Literal[-1, False, 1]: ...  # 检查两个向量是否平行的函数签名

def get_parallels(
    bezier2: ArrayLike, width: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]: ...  # 获取与给定贝塞尔曲线平行的点对的函数签名

def find_control_points(
    c1x: float, c1y: float, mmx: float, mmy: float, c2x: float, c2y: float
) -> list[tuple[float, float]]: ...  # 查找控制点的函数签名

def make_wedged_bezier2(
    bezier2: ArrayLike, width: float, w1: float = ..., wm: float = ..., w2: float = ...
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]: ...  # 创建具有楔形特性的贝塞尔曲线的函数签名
```